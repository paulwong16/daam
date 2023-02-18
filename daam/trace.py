from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Union
import math

from diffusers import StableDiffusionPipeline
from diffusers.models.attention import CrossAttention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F

from .utils import cache_dir, auto_autocast, compute_token_merge_indices
from .experiment import GenerationExperiment
from .heatmap import RawHeatMapCollection, GlobalHeatMap
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator


__all__ = ['trace', 'DiffusionHeatMapHooker', 'GlobalHeatMap']


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(
            self,
            pipeline:
            StableDiffusionPipeline,
            low_memory: bool = False,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: str = None,
            prompt: str = None,
            highlight_key_words: list = None, # [Focus_more_keywords, Focus_less_keywords]. String separate by space.
            highlight_amp_mags: list = None # [more_magnitude, less_magnitude] hightlight amplified magnitude 
            
    ):
        self.all_heat_maps = RawHeatMapCollection()
        h = (pipeline.unet.config.sample_size * pipeline.vae_scale_factor)
        self.latent_hw = 4096 if h == 512 else 9216  # 64x64 or 96x96 depending on if it's 2.0-v or 2.0
        locate_middle = load_heads or save_heads
        self.locator = UNetCrossAttentionLocator(restrict={0} if low_memory else None, locate_middle_block=locate_middle)
        self.last_prompt: str = ''
        self.last_image: Image = None
        self.time_idx = 0
        self._gen_idx = 0
        self.user_input_prompt = prompt
        self.user_hightlight_key_words = highlight_key_words
        if self.user_hightlight_key_words is not None:
            print("Hightlight keywords: {}".format(self.user_hightlight_key_words))
            if highlight_key_words[0] is not None:
                focus_more = compute_token_merge_indices(pipeline.tokenizer, self.user_input_prompt,
                                                                highlight_key_words[0])[0]
            else:
                focus_more = None
            if highlight_key_words[1] is not None:
                focus_less = compute_token_merge_indices(pipeline.tokenizer, self.user_input_prompt,
                                                                highlight_key_words[1])[0]
            else:
                focus_less = None
            self.user_hightlight_key_words = [focus_more, focus_less]
            print("Hightlight index: {}".format(self.user_hightlight_key_words))
        self.user_highlight_amp_mags = highlight_amp_mags

        modules = [
            UNetCrossAttentionHooker(
                x,
                self,
                layer_idx=idx,
                latent_hw=self.latent_hw,
                load_heads=load_heads,
                save_heads=save_heads,
                data_dir=data_dir
            ) for idx, x in enumerate(self.locator.locate(pipeline.unet))
        ]

        modules.append(PipelineHooker(pipeline, self))

        super().__init__(modules)
        self.pipe = pipeline

    def time_callback(self, *args, **kwargs):
        self.time_idx += 1

    @property
    def layer_names(self):
        return self.locator.layer_names

    def to_experiment(self, path, seed=None, id='.', subtype='.', **compute_kwargs):
        # type: (Union[Path, str], int, str, str, Dict[str, Any]) -> GenerationExperiment
        """Exports the last generation call to a serializable generation experiment."""

        return GenerationExperiment(
            self.last_image,
            self.compute_global_heat_map(**compute_kwargs).heat_maps,
            self.last_prompt,
            seed=seed,
            id=id,
            subtype=subtype,
            path=path,
            tokenizer=self.pipe.tokenizer,
        )

    def compute_global_heat_map(self, prompt=None, factors=None, head_idx=None, layer_idx=None, normalize=False):
        # type: (str, List[float], int, int, bool) -> GlobalHeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.all_heat_maps

        if prompt is None:
            prompt = self.last_prompt

        if factors is None:
            factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)

        all_merges = []
        x = int(np.sqrt(self.latent_hw))

        with auto_autocast(dtype=torch.float32):
            for (factor, layer, head), heat_map in heat_maps:
                if factor in factors and (head_idx is None or head_idx == head) and (layer_idx is None or layer_idx == layer):
                    heat_map = heat_map.unsqueeze(1)
                    # The clamping fixes undershoot.
                    all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

            try:
                maps = torch.stack(all_merges, dim=0)
            except RuntimeError:
                if head_idx is not None or layer_idx is not None:
                    raise RuntimeError('No heat maps found for the given parameters.')
                else:
                    raise RuntimeError('No heat maps found. Did you forget to call `with trace(...)` during generation?')

            maps = maps.mean(0)[:, 0]
            maps = maps[:len(self.pipe.tokenizer.tokenize(prompt)) + 2]  # 1 for SOS and 1 for padding

            if normalize:
                maps = maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities

        return GlobalHeatMap(self.pipe.tokenizer, prompt, maps)


class PipelineHooker(ObjectHooker[StableDiffusionPipeline]):
    def __init__(self, pipeline: StableDiffusionPipeline, parent_trace: 'trace'):
        super().__init__(pipeline)
        self.heat_maps = parent_trace.all_heat_maps
        self.parent_trace = parent_trace

    def _hooked_run_safety_checker(hk_self, self: StableDiffusionPipeline, image, *args, **kwargs):
        image, has_nsfw = hk_self.monkey_super('run_safety_checker', image, *args, **kwargs)
        pil_image = self.numpy_to_pil(image)
        hk_self.parent_trace.last_image = pil_image[0]

        return image, has_nsfw

    def _hooked_encode_prompt(hk_self, _: StableDiffusionPipeline, prompt: Union[str, List[str]], *args, **kwargs):
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError('Only single prompt generation is supported for heat map computation.')
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt

        hk_self.heat_maps.clear()
        hk_self.parent_trace.last_prompt = last_prompt
        ret = hk_self.monkey_super('_encode_prompt', prompt, *args, **kwargs)

        return ret

    def _hook_impl(self):
        self.monkey_patch('run_safety_checker', self._hooked_run_safety_checker)
        self.monkey_patch('_encode_prompt', self._hooked_encode_prompt)


class UNetCrossAttentionHooker(ObjectHooker[CrossAttention]):
    def __init__(
            self,
            module: CrossAttention,
            parent_trace: 'trace',
            context_size: int = 77,
            layer_idx: int = 0,
            latent_hw: int = 9216,
            load_heads: bool = False,
            save_heads: bool = False,
            data_dir: Union[str, Path] = None,
    ):
        super().__init__(module)
        self.heat_maps = parent_trace.all_heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

        self.load_heads = load_heads
        self.save_heads = save_heads
        self.trace = parent_trace

        if data_dir is not None:
            data_dir = Path(data_dir)
        else:
            data_dir = cache_dir() / 'heads'

        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with auto_autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        return maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)
    
    def _amplifier(self, input_x):
        if self.trace.user_hightlight_key_words is None:
            return input_x
        multiplier = torch.ones_like(input_x)
        focus_more = self.trace.user_hightlight_key_words[0]
        focus_less = self.trace.user_hightlight_key_words[1]
        if focus_more is not None:
            multiplier[..., focus_more] = self.trace.user_highlight_amp_mags[0]
        if focus_less is not None:
            multiplier[..., focus_less] = self.trace.user_highlight_amp_mags[1]
        return input_x * multiplier

    def _hooked_sliced_attention(hk_self, self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query[start_idx:end_idx],
                key[start_idx:end_idx].transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            attn_slice = attn_slice.softmax(dim=-1)
            factor = int(math.sqrt(hk_self.latent_hw // attn_slice.shape[1]))

            if attn_slice.shape[-1] == hk_self.context_size:
                # shape: (batch_size, 64 // factor, 64 // factor, 77)
                maps = hk_self._unravel_attn(attn_slice)

                for head_idx, heatmap in enumerate(maps):
                    hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _save_attn(self, attn_slice: torch.Tensor):
        torch.save(attn_slice, self.data_dir / f'{self.trace._gen_idx}.pt')

    def _load_attn(self) -> torch.Tensor:
        return torch.load(self.data_dir / f'{self.trace._gen_idx}.pt')

    def _hooked_get_attn_score(hk_self, self, query, key, attention_mask=None):
        """
        Monkey-patched version of :py:func:`.CrossAttention.get_attention_scores` to capture attentions and aggregate them.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1
            
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        
        factor = int(math.sqrt(hk_self.latent_hw // attention_scores.shape[1]))
        if attention_scores.shape[-1] == hk_self.context_size and factor != 8:
            attention_scores = hk_self._amplifier(attention_scores)
        attention_probs = attention_scores.softmax(dim=-1)

        if hk_self.save_heads:
            hk_self._save_attn(attention_probs)
        elif hk_self.load_heads:
            attention_probs = hk_self._load_attn()
        
        hk_self.trace._gen_idx += 1
        if attention_probs.shape[-1] == hk_self.context_size and factor != 8:
            # shape: (batch_size, 64 // factor, 64 // factor, 77)
            maps = hk_self._unravel_attn(attention_probs)

            for head_idx, heatmap in enumerate(maps):
                hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap)
        attention_probs = attention_probs.to(dtype)
            
        return attention_probs
        
        
    def _hooked_attention(hk_self, self, query, key, value):
        """
        Monkey-patched version of :py:func:`.CrossAttention._attention` to capture attentions and aggregate them.

        Args:
            hk_self (`UNetCrossAttentionHooker`): pointer to the hook itself.
            self (`CrossAttention`): pointer to the module.
            query (`torch.Tensor`): the query tensor.
            key (`torch.Tensor`): the key tensor.
            value (`torch.Tensor`): the value tensor.
        """

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attn_slice = attention_scores.softmax(dim=-1)

        if hk_self.save_heads:
            hk_self._save_attn(attn_slice)
        elif hk_self.load_heads:
            attn_slice = hk_self._load_attn()

        factor = int(math.sqrt(hk_self.latent_hw // attn_slice.shape[1]))
        hk_self.trace._gen_idx += 1

        if attn_slice.shape[-1] == hk_self.context_size and factor != 8:
            # shape: (batch_size, 64 // factor, 64 // factor, 77)
            maps = hk_self._unravel_attn(attn_slice)

            for head_idx, heatmap in enumerate(maps):
                hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap)

        # compute attention output
        hidden_states = torch.bmm(attn_slice, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _hook_impl(self):
        self.monkey_patch('get_attention_scores', self._hooked_get_attn_score)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
