import torch
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from difflib import SequenceMatcher
import random


class SimpleDiff:
    def __init__(self, device, unet, vae):
        self.device = device
        self.unet = unet
        self.vae = vae

    def load(self):
        self.unet.to(self.device)
        self.vae.to(self.device)

    #Setup UNet weights
    def init_attention_weights(self, weight_tuples, tokens_length):
        weights = torch.ones(tokens_length)

        for i, w in weight_tuples:
            if i < tokens_length and i >= 0:
                weights[i] = w

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.last_attn_slice_weights = weights.to(self.device)
            if module_name == "CrossAttention" and "attn1" in name:
                module.last_attn_slice_weights = None

    #Setup Unet Edit
    def init_attention_edit(self, tokens, tokens_edit, tokens_length):
        mask = torch.zeros(tokens_length)
        indices_target = torch.arange(tokens_length, dtype=torch.long)
        indices = torch.zeros(tokens_length, dtype=torch.long)

        tokens = tokens.input_ids.numpy()[0]
        tokens_edit = tokens_edit.input_ids.numpy()[0]

        for name, a0, a1, b0, b1 in SequenceMatcher(None, tokens, tokens_edit).get_opcodes():
            if b0 < tokens_length:
                if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                    mask[b0:b1] = 1
                    indices[b0:b1] = indices_target[a0:a1]

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.last_attn_slice_mask = mask.to(self.device)
                module.last_attn_slice_indices = indices.to(self.device)
            if module_name == "CrossAttention" and "attn1" in name:
                module.last_attn_slice_mask = None
                module.last_attn_slice_indices = None

    #Setup Unet Funcs
    def init_attention_func(self):
        #ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276
        def new_attention(self, query, key, value):
            # TODO: use baddbmm for better performance
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            attn_slice = attention_scores.softmax(dim=-1)
            # compute attention output

            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice

                self.use_last_attn_slice = False

            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False

            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False

            hidden_states = torch.matmul(attn_slice, value)
            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        def new_sliced_attention(self, query, key, value, sequence_length, dim):

            batch_size_attention = query.shape[0]
            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
            )
            slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
            for i in range(hidden_states.shape[0] // slice_size):
                start_idx = i * slice_size
                end_idx = (i + 1) * slice_size
                attn_slice = (
                        torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
                )  # TODO: use baddbmm for better performance
                attn_slice = attn_slice.softmax(dim=-1)

                if self.use_last_attn_slice:
                    if self.last_attn_slice_mask is not None:
                        new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                        attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                    else:
                        attn_slice = self.last_attn_slice

                    self.use_last_attn_slice = False

                if self.save_last_attn_slice:
                    self.last_attn_slice = attn_slice
                    self.save_last_attn_slice = False

                if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                    attn_slice = attn_slice * self.last_attn_slice_weights
                    self.use_last_attn_weights = False

                attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

                hidden_states[start_idx:end_idx] = attn_slice

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention":
                module.last_attn_slice = None
                module.use_last_attn_slice = False
                module.use_last_attn_weights = False
                module.save_last_attn_slice = False
                module._sliced_attention = new_sliced_attention.__get__(module, type(module))
                module._attention = new_attention.__get__(module, type(module))

    def use_last_tokens_attention(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.use_last_attn_slice = use

    def use_last_tokens_attention_weights(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.use_last_attn_weights = use

    def use_last_self_attention(self, use=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn1" in name:
                module.use_last_attn_slice = use

    def save_last_tokens_attention(self, save=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.save_last_attn_slice = save

    def save_last_self_attention(self, save=True):
        for name, module in self.unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn1" in name:
                module.save_last_attn_slice = save

    @torch.no_grad()
    def generate(self, embedding_unconditional,
                 embedding_conditional,
                 tokens_conditional_edit=None, embedding_conditional_edit=None,
                 tokens_length=77, tokens_conditional=None, init_latents=None,  guidance_scale=7.5, steps=50, seed=None, width=512, height=512,
                 prompt_edit=None, prompt_edit_token_weights=[], prompt_edit_tokens_start=0.0,
                 prompt_edit_tokens_end=1.0, prompt_edit_spatial_start=0.0, prompt_edit_spatial_end=1.0):
        print("generating..")

        #Change size to multiple of 64 to prevent size mismatches inside model
        if init_latents is not None:
            width = init_latents.shape[-1] * 8
            height = init_latents.shape[-2] * 8

        width = width - width % 64
        height = height - height % 64

        #If seed is None, randomly select seed from 0 to 2^32-1
        if seed is None: seed = random.randrange(2**32 - 1)
        generator = torch.cuda.manual_seed(seed)

        #Set inference timesteps to scheduler
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(steps)

        init_latent = torch.zeros((1, self.unet.in_channels, height // 8, width // 8), device=self.device)
        t_start = 0

        #Generate random normal noise
        noise = torch.randn(init_latent.shape, generator=generator, device=self.device)

        #If init_latents is used, initialize noise as init_latent
        if init_latents is not None:
            noise = init_latents

        init_latents = noise
        latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=self.device)).to(self.device)

        #Process clip
        with autocast("cuda"):
            #TODO
            #tokens_unconditional = clip_tokenizer("", padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            #embedding_unconditional = clip(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

            #tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            #embedding_conditional = clip(tokens_conditional.input_ids.to(self.device)).last_hidden_state

            #Process prompt editing
            if prompt_edit is not None:
                #tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                #embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(self.device)).last_hidden_state

                self.init_attention_edit(tokens_conditional, tokens_conditional_edit, tokens_length)

            self.init_attention_func()
            self.init_attention_weights(prompt_edit_token_weights, tokens_length)

            timesteps = scheduler.timesteps[t_start:]

            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                #sigma = scheduler.sigmas[t_index]
                latent_model_input = latent
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                #Predict the unconditional noise residual
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample

                #Prepare the Cross-Attention layers
                if prompt_edit is not None:
                    self.save_last_tokens_attention()
                    self.save_last_self_attention()
                else:
                    #Use weights on non-edited prompt when edit is None
                    self.use_last_tokens_attention_weights()

                #Predict the conditional noise residual and save the cross-attention layer activations
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample

                #Edit the Cross-Attention layer activations
                if prompt_edit is not None:
                    t_scale = t / scheduler.num_train_timesteps
                    if prompt_edit_tokens_start <= t_scale <= prompt_edit_tokens_end:
                        self.use_last_tokens_attention()
                    if prompt_edit_spatial_start <= t_scale <= prompt_edit_spatial_end:
                        self.use_last_self_attention()

                    #Use weights on edited prompt
                    self.use_last_tokens_attention_weights()

                    #Predict the edited conditional noise residual using the cross-attention masks
                    noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample

                #Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latent = scheduler.step(noise_pred, t_index, latent).prev_sample

        #scale and decode the image latents with vae
        latent = latent / 0.18215
        image = self.vae.decode(latent.to(self.vae.dtype)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype("uint8")
        return Image.fromarray(image)



class CLIP:
    def __init__(self, device, clip_model, clip_processor):
        self.device = device
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.max_length = 77

    def load(self):
        self.clip_model.to(self.device)

    def tokenize(self, texts, padding="max_length", truncation=True, return_tensors="pt",
                 return_overflowing_tokens=True):
        return self.clip_processor(text=texts, return_tensors=return_tensors, padding=padding,
                                   truncation=truncation, return_overflowing_tokens=return_overflowing_tokens)
        # return self.clip_tokenizer(prompt, padding=padding, max_length=self.max_length,
        #                            truncation=truncation, return_tensors=return_tensors,
        #                            return_overflowing_tokens=return_overflowing_tokens)

    def embed_text(self, texts):
        # return self.clip_model.text_model(tokens.input_ids.to(self.device)).pooler_output
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding="max_length", truncation=True, return_overflowing_tokens=True)
        inputs.to(self.device)
        with autocast("cuda"):
            # attention_mask must be disregarded
            return self.clip_model.text_model(input_ids=inputs.input_ids).last_hidden_state

    def pooled_text(self, texts):
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)
        inputs.to(self.device)
        with autocast("cuda"):
            return self.clip_model.text_projection(self.clip_model.text_model(**inputs)[1])

    def embed_image(self, images):
        inputs = self.clip_processor(images=images, return_tensors="pt")
        inputs.to(self.device)
        with autocast("cuda"):
            # return self.clip_model.visual_projection(self.clip_model.vision_model(**inputs)[1]) #TODO
            # print(inputs)
            return self.clip_model.visual_projection(self.clip_model.vision_model(pixel_values=inputs.pixel_values).last_hidden_state[:,:77])


