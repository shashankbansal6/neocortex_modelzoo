# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from typing import Tuple

import torch

from modelzoo.common.pytorch.model_utils.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)


class Converter_GPTJ_Attention_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("q_proj", "proj_q_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("k_proj", "proj_k_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("v_proj", "proj_v_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("out_proj", "proj_output_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                # This is a hacky way to initialize bias and masked_bias when converting from CS to Huggingface
                # However, based on huggingface implementation this is unavoidable in order to initialize
                # attn.bias and attn.masked_bias, which we intiantiate when we capture the `out_proj` key
                action=self.replace_or_fill_masked_bias,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None

    def replace_or_fill_masked_bias(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # copy between `out_proj` and `proj_output_dense_layer`
        new_state_dict[new_key] = old_state_dict[old_key]

        # take care of bias and masked_bias
        if from_index == 1:
            max_positions = action_fn_args["configs"][1]["model"][
                "max_position_embeddings"
            ]
            bias_key = re.sub("out_proj\.weight", "bias", new_key)
            masked_bias_key = re.sub("out_proj\.weight", "masked_bias", new_key)
            new_state_dict[bias_key] = torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.uint8)
            ).view(1, 1, max_positions, max_positions)
            new_state_dict[masked_bias_key] = torch.tensor(-1e9)


class Converter_GPTJ_Headless_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # word embeddings
            ConversionRule(
                [
                    EquivalentSubkey("wte", "embedding_layer.word_embeddings"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # final layer norm
            ConversionRule(
                [
                    EquivalentSubkey("ln_f", "transformer_decoder.norm"),
                    "\.(?:weight|bias)",
                ],
                action=self.replace_final_norm,
            ),
            # attention
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("attn.", "self_attn."),
                    Converter_GPTJ_Attention_HF_CS17(),
                ],
                action=None,
            ),
            # attention norm
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("ln_1", "norm1"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # norm3 layers should be set to None if they exist:
            ConversionRule(
                ["transformer_decoder.layers\.\d+\.norm3\.(?:weight|bias)",],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, None),
            ),
            # intermediate ffn
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("mlp.fc_in", "ffn.ffn.0.linear_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("mlp.fc_out", "ffn.ffn.1.linear_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(["lm_head\.(?:weight|bias)"], exists="right"),
            ConversionRule(["ln_f\.(?:weight|bias)"], exists="right"),
            ConversionRule(["h\.\d+\.attn\.(?:masked_bias|bias)",]),
        ]

    def replace_final_norm(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        new_state_dict[new_key] = old_state_dict[old_key]
        # CS 1.7 has both "ln_f" and "transformer_decoder.norm"
        # we need to copy the original ("ln_f") too:
        if from_index == 0:
            ln_f_key = re.sub("transformer_decoder\.norm\.", "ln_f.", new_key)
            new_state_dict[ln_f_key] = old_state_dict[old_key]

    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            logging.warning(
                "{} GPTJ has a language model head (lm_head) "
                "while {} GPTJModel does not. Initializing lm_head to default."
            )

        # Manually tie weights
        if from_index == 1 and configs[1]["model"]["share_embedding_weights"]:
            if (
                old_state_dict.get("embedding_layer.word_embeddings.weight", 0)
                is None
            ):
                old_state_dict[
                    "embedding_layer.word_embeddings.weight"
                ] = old_state_dict["lm_head.weight"]

    def post_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            # We are converting from HF GPTJModel (which is headless) -> CS GPTJModel (which has a head)
            # We need to create 'lm_head' and init to default values
            hf_config = configs[0]
            cs_config = configs[1]
            use_bias_in_output = cs_config["model"].get(
                "use_bias_in_output", False
            )
            vocab_size = cs_config["model"]["vocab_size"]
            embed_dim = cs_config["model"]["hidden_size"]
            if hf_config["tie_word_embeddings"]:
                lm_head_weight = old_state_dict['wte.weight']
            else:
                lm_head_weight = torch.zeros((vocab_size, embed_dim))
                lm_head_weight.normal_(mean=0.0, std=0.02)
            new_state_dict["lm_head.weight"] = lm_head_weight
            if use_bias_in_output:
                lm_head_bias = torch.zeros(vocab_size)
                new_state_dict["lm_head.bias"] = lm_head_bias

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} GPTJModel <-> {} GPTJModel(with head)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS17


class Converter_GPTJ_Headless_HF_CS18(Converter_GPTJ_Headless_HF_CS17):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule([Converter_GPTJ_Headless_HF_CS17(),], action=None,),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_GPTJ_Headless_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} GPTJModel <-> {} GPTJModel(with head)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS18


class Converter_GPTJ_LMHeadModel_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["lm_head\.(?:weight|bias)"], action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("transformer.", ""),
                    Converter_GPTJ_Headless_HF_CS17(),
                ],
                action=None,
            ),
        ]

    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        from_index,
        drop_unmatched_keys,
    ):
        # Manually tie weights
        if from_index == 1 and configs[1]["model"]["share_embedding_weights"]:
            if (
                old_state_dict.get("embedding_layer.word_embeddings.weight", 0)
                is None
            ):
                old_state_dict[
                    "embedding_layer.word_embeddings.weight"
                ] = old_state_dict["lm_head.weight"]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPTJForCausalLM <-> {} GPTJModel(with head)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS17


class Converter_GPTJ_LMHeadModel_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [Converter_GPTJ_LMHeadModel_HF_CS17(),], action=None,
            ),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_GPTJ_LMHeadModel_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPTJForCausalLM <-> {} GPTJModel(with head)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS18


class ConfigConverter_GPTJModel_HF_CS17(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(["rotary_dim"], action=self.replaceKey),
            ConversionRule(
                [EquivalentSubkey("rotary", "position_embedding_type")],
                exists="right",
                action=self.convert_position_embedding_type,
            ),
            ConversionRule(
                ["use_position_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                [EquivalentSubkey("embd_pdrop", "embedding_dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "tie_word_embeddings", "share_embedding_weights"
                    )
                ],
                action=self.replaceKey,
            ),
            # Decoder Block
            ConversionRule(
                [EquivalentSubkey("n_embd", "hidden_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_head", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_layer", "num_hidden_layers")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_positions", "max_position_embeddings")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["scale_attn_weights"],
                action=BaseConfigConverter.assert_factory_fn(0, True),
            ),
            ConversionRule(
                ["attention_type"],
                action=BaseConfigConverter.assert_factory_fn(
                    1, "scaled_dot_product"
                ),
            ),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["use_ffn_bias_in_attention"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["use_ffn_bias"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                [EquivalentSubkey("n_inner", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("activation_function", "nonlinearity")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("attn_pdrop", "attention_dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("resid_pdrop", "residual_dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(["layer_norm_epsilon"], action=self.replaceKey,),
            ConversionRule(["use_bias_in_output"], action=self.replaceKey,),
            ConversionRule(["initializer_range"], action=self.replaceKey),
            ConversionRule(
                ["embedding_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["fixed_sparse_attention"],
                action=BaseConfigConverter.assert_factory_fn(1, None),
            ),
            ConversionRule(
                ["norm_first"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_ff_layer1_dropout"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
        ]

    def convert_position_embedding_type(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            if old_state_dict[old_key] != True:
                raise ConfigConversionError(
                    "HF GPT-J must use rotary embeddings, but got {}={}".format(
                        old_key, old_state_dict[old_key]
                    )
                )
            new_state_dict[new_key] = "rotary"
        else:
            if old_state_dict[old_key] != "rotary":
                raise ConfigConversionError(
                    "CS GPT-J must use rotary embeddings, but got {}={}".format(
                        old_key, old_state_dict[old_key]
                    )
                )
            new_state_dict[new_key] = True

    def convert_attention_type(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            new_state_dict[new_key] = (
                "scaled_dot_product"
                if old_state_dict[old_key]
                else "dot_product"
            )
        else:
            assert (
                old_state_dict[old_key] == "scaled_dot_product"
                or old_state_dict[old_key] == "dot_product"
            )
            new_state_dict[new_key] = old_state_dict[old_key].startswith(
                "scaled_"
            )

    def pre_config_convert(
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)

        defaults = [
            {
                "vocab_size": 50400,
                "n_positions": 2048,
                "n_embd": 4096,
                "n_layer": 28,
                "n_head": 16,
                "rotary_dim": 64,
                "activation_function": "gelu_new",
                "resid_pdrop": 0.1,
                "embd_pdrop": 0.1,
                "attn_pdrop": 0.1,
                "initializer_range": 0.02,
                "layer_norm_epsilon": 1e-5,
                "tie_word_embeddings": False,
            },
            {
                "max_position_embeddings": 1024,
                "embedding_dropout_rate": 0.1,
                "share_embedding_weights": True,
                "residual_dropout_rate": 0.1,
                "nonlinearity": "gelu",
                "layer_norm_epsilon": 1.0e-5,
                "use_ffn_bias": False,
                "use_untied_layer_norm": False,
                "attention_dropout_rate": 0.1,
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "initializer_range": 0.02,
                "use_bias_in_output": False,
                "norm_first": True,
            },
        ]

        # Apply defaults
        for key in defaults[from_index]:
            if key not in config:
                config[key] = defaults[from_index][key]

        if from_index == 0:
            if "n_inner" not in config or config["n_inner"] is None:
                config["n_inner"] = 4 * config["n_embd"]
        return config

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            new_config["use_ffn_bias_in_attention"] = False
            new_config["use_projection_bias_in_attention"] = False
            new_config["use_ffn_bias"] = True
            new_config["use_bias_in_output"] = True
            if "attention_type" not in new_config:
                new_config["attention_type"] = "scaled_dot_product"

        return super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


class ConfigConverter_GPTJModel_HF_CS18(ConfigConverter_GPTJModel_HF_CS17):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))
