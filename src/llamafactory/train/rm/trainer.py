# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
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

import json
import os
import sys
import random
from types import MethodType
from typing import TYPE_CHECKING, Optional, Union, Dict, Any, List

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


from .loss import SimilarityLoss

# Initialize logger
logger = logging.get_logger(__name__)

# Add rankings file loading
_RANKINGS_CACHE = {}
_ADS_BY_ID_CACHE = {}
_QUERY_RESPONSES_CACHE = {}
_CLASSIFIED_ADS_CACHE = {}

def load_rankings_file(rankings_path="src/llamafactory/train/rm/rankings_original.json"):
    """Load rankings data from a JSON file."""
    if rankings_path not in _RANKINGS_CACHE:
        try:
            with open(rankings_path, 'r', encoding='utf-8') as f:
                rankings_data = json.load(f)
                
            # Transform into a more convenient format for lookups
            transformed = {}
            for item in rankings_data:
                query = item.get("query", {}).get("query", "")
                if query:
                    transformed[query] = item
            
            _RANKINGS_CACHE[rankings_path] = transformed
            return transformed
        except Exception as e:
            logger.warning(f"Failed to load rankings file: {str(e)}")
            return {}
    return _RANKINGS_CACHE[rankings_path]

def load_ads_by_id(ads_path="src/llamafactory/train/rm/200_sampled_ads.json"):
    """Load the full ad content by ID."""
    if ads_path not in _ADS_BY_ID_CACHE:
        try:
            with open(ads_path, 'r', encoding='utf-8') as f:
                ads_data = json.load(f)
            
            # Create a lookup dictionary by ad_id
            ads_by_id = {ad["ad_id"]: ad for ad in ads_data if "ad_id" in ad}
            _ADS_BY_ID_CACHE[ads_path] = ads_by_id
            return ads_by_id
        except Exception as e:
            logger.warning(f"Failed to load ads file: {str(e)}")
            return {}
    return _ADS_BY_ID_CACHE[ads_path]

def load_classified_ads(path="src/llamafactory/train/rm/200_classified_ads.json"):
    """Load the classified ads data."""
    if path not in _CLASSIFIED_ADS_CACHE:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create lookup by ad_id
            classified_ads = {item["id"]: {"domain": item["domain"], "subdomain": item["subdomain"]} 
                            for item in data}
            _CLASSIFIED_ADS_CACHE[path] = classified_ads
            return classified_ads
        except Exception as e:
            logger.warning(f"Failed to load classified ads: {str(e)}")
            return {}
    return _CLASSIFIED_ADS_CACHE[path]

def load_query_responses(path="src/llamafactory/train/rm/query_responses_original_200.json"):
    """Load query responses data."""
    if path not in _QUERY_RESPONSES_CACHE:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create lookup by query text
            query_responses = {}
            for item in data:
                query = item.get("query", "")
                if query:
                    query_responses[query] = {
                        "domain": item.get("domain", ""),
                        "subdomain": item.get("subdomain", ""),
                        "documents": item.get("documents_in_response", [])
                    }
            
            _QUERY_RESPONSES_CACHE[path] = query_responses
            return query_responses
        except Exception as e:
            logger.warning(f"Failed to load query responses: {str(e)}")
            return {}
    return _QUERY_RESPONSES_CACHE[path]

class CustomRewardModel(torch.nn.Module): 
    def __init__(self, raw_ads, classified_ads, original_responses, similarity_loss_fn):
        super().__init__()
        self.raw_ads = raw_ads
        self.classified_ads = classified_ads
        self.original_responses = original_responses
        self.similarity_loss_fn = similarity_loss_fn

    def get_rewards(self, decoded_responses, top_k_docs_dict=None):
        """
        Calculate rewards for decoded responses based on matched queries and top-k docs
        
        Args:
            decoded_responses: List of decoded text responses
            top_k_docs_dict: Dictionary mapping queries to their top-k docs
            
        Returns:
            Scalar tensor reward that can be used for gradient computation
        """
        total_losses = []
        
        if top_k_docs_dict is None:
            # If no top_k_docs provided, we'll build them on demand
            rankings = load_rankings_file()
            ads_by_id = load_ads_by_id()
            top_k_docs_dict = {}
        
        for ad_idx, original_ad in enumerate(self.raw_ads):
            if ad_idx >= len(decoded_responses):
                # Skip if we don't have a corresponding rewritten ad
                continue
                
            rewritten_text = decoded_responses[ad_idx]
            
            #add the metadata from the original ad to the rewritten ad
            if isinstance(rewritten_text, str):
                rewritten_ad = original_ad.copy()
                rewritten_ad["text"] = rewritten_text
            else:
                rewritten_ad = rewritten_text
                
            ad_id = original_ad.get("ad_id", "")
            if not ad_id or ad_id not in self.classified_ads:
                # Skip ads we can't classify
                continue
                
            # Get ad domain/subdomain
            ad_domain = self.classified_ads.get(ad_id, {}).get("domain", "")
            ad_subdomain = self.classified_ads.get(ad_id, {}).get("subdomain", "")
            
            if not ad_domain or not ad_subdomain:
                continue
                
            # Find relevant queries for this ad
            relevant_queries = []
            for query, q_info in self.original_responses.items():
                if q_info.get("domain") == ad_domain and q_info.get("subdomain") == ad_subdomain:
                    relevant_queries.append(query)
            
            if not relevant_queries:
                continue
                
            # Calculate loss for relevant queries
            losses = []
            sample_size = min(len(relevant_queries), 3)  # Limit to 3 queries for efficiency
            for query in random.sample(relevant_queries, sample_size):
                # Get or build top_k docs for this query
                if query in top_k_docs_dict:
                    docs_for_query = top_k_docs_dict[query]
                else:
                    docs_for_query = build_top_k_docs(query)
                    top_k_docs_dict[query] = docs_for_query
                
                # Calculate loss for this query using SimilarityLoss
                loss = self.similarity_loss_fn(
                    query, 
                    original_ad.get("text", ""), 
                    rewritten_ad.get("text", ""), 
                    docs_for_query
                )
                
                # Ensure loss is scalar
                if loss.dim() > 0 and loss.numel() > 1:
                    loss = loss.mean()
                
                # Create a dummy connection to ensure gradient flow if needed
                if not loss.requires_grad:
                    # A small value multiplied by a requires_grad tensor ensures gradient flow
                    dummy = torch.tensor(1e-10, requires_grad=True)
                    loss = loss + dummy
                
                # Add to list of losses
                losses.append(loss)
            
            # Average the losses for this ad across different queries
            if losses:
                # Simply mean the losses (they're all scalars now)
                avg_loss = sum(losses) / len(losses)
                total_losses.append(avg_loss)
        
        # Calculate final reward as the negative of loss (higher reward is better)
        if total_losses:
            # Always return a scalar reward
            final_reward = -sum(total_losses) / len(total_losses)
            return final_reward  # Negative because lower loss = higher reward
        else:
            # Return default tensor with grad enabled
            # Using 0.0 as neutral reward when we can't calculate
            return torch.tensor(0.0, requires_grad=True)

def build_top_k_docs(query, rankings=None, original_ads_by_id=None, max_docs=5):
    """
    Build a list of top-k documents for a given query from rankings data.
    
    Args:
        query: The query text to look up
        rankings: Dictionary mapping queries to ranking information
        original_ads_by_id: Dictionary mapping ad IDs to ad content
        max_docs: Maximum number of documents to include
        
    Returns:
        List of documents (text strings) that are top ranked for the query
    """
    if rankings is None:
        rankings = load_rankings_file()
    
    if original_ads_by_id is None:
        original_ads_by_id = load_ads_by_id()
    
    # Find ranking info for this query
    ranking_info = None
    for query_text, query_info in rankings.items():
        # Exact match or substring match
        if query == query_text or query in query_text or query_text in query:
            ranking_info = query_info
            break
    
    if not ranking_info:
        return []
    
    # Get ranked ad IDs
    ranked_ad_ids = ranking_info.get('ranked_ad_ids', [])
    if not ranked_ad_ids:
        return []
    
    # Limit to max_docs
    ranked_ad_ids = ranked_ad_ids[:max_docs]
    
    # Convert to document texts
    docs = []
    for ad_id in ranked_ad_ids:
        if ad_id in original_ads_by_id:
            ad = original_ads_by_id[ad_id]
            # Extract just the text or use the whole ad object depending on what's needed
            if isinstance(ad, dict) and 'text' in ad:
                docs.append(ad['text'])
            else:
                docs.append(str(ad))
    
    return docs

class PairwiseTrainer(Trainer):
    r"""Inherits Trainer to compute pairwise loss."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
            
        # Add find_unused_parameters=True to fix distributed training with custom loss
        if kwargs.get("args") is not None:
            # Set the find_unused_parameters flag directly on the training args
            training_args = kwargs.get("args")
            if hasattr(training_args, "ddp_find_unused_parameters"):
                training_args.ddp_find_unused_parameters = True
            elif hasattr(training_args, "_n_gpu") and training_args._n_gpu > 1:
                logger.info_rank0("Setting find_unused_parameters=True for distributed training with custom loss")
                # Try to add it as an attribute if it doesn't exist
                setattr(training_args, "ddp_find_unused_parameters", True)

        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss
        self.add_callback(FixValueHeadModelCallback)
        
        # Initialize the custom SimilarityLoss with hard-coded parameters
        # Set this to True to use your custom loss, False to use standard pairwise loss
        self.use_custom_loss = True  # Change to False to disable
        self.similarity_loss = None
        self.custom_reward_model = None
        
        if self.use_custom_loss:
            try:
                # Hard-coded parameters for SimilarityLoss
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                alpha = 1.0  # Weight for L1 loss (query-document similarity)
                beta = 1.0   # Weight for L2 loss (triplet sampling)
                gamma = 1.0  # Weight for L3 loss (preservation)
                
                self.similarity_loss = SimilarityLoss(
                    embedding_model_name=embedding_model,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )
                logger.info_rank0(f"Initialized SimilarityLoss with α={alpha}, β={beta}, γ={gamma}")
                
                # Initialize the CustomRewardModel
                raw_ads = list(load_ads_by_id().values())
                classified_ads = load_classified_ads()
                original_responses = load_query_responses()
                
                if raw_ads and classified_ads and original_responses:
                    self.custom_reward_model = CustomRewardModel(
                        raw_ads=raw_ads,
                        classified_ads=classified_ads,
                        original_responses=original_responses,
                        similarity_loss_fn=self.similarity_loss
                    )
                    logger.info_rank0("Successfully initialized CustomRewardModel")
                else:
                    logger.warning("Could not initialize CustomRewardModel, missing data")
                
            except Exception as e:
                logger.warning(f"Failed to initialize SimilarityLoss: {str(e)}")
                self.use_custom_loss = False

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        # First call the parent class method to setup the optimizer properly
        optimizer = super().create_optimizer()
        # Unconditionally disable gradient checkpointing for custom loss training
        if self.use_custom_loss:
            # Disable checkpointing on the main model
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                logger.info_rank0("Disabling gradient checkpointing on main model for custom loss training")
                self.model.gradient_checkpointing_disable()
            # Clear config flag if present
            if hasattr(self.model.config, 'gradient_checkpointing'):
                self.model.config.gradient_checkpointing = False
            # Disable on distributed module wrapper
            if hasattr(self.model, 'module') and hasattr(self.model.module, 'gradient_checkpointing_disable'):
                self.model.module.gradient_checkpointing_disable()
            # Disable checkpointing for similarity loss encoder (SentenceTransformer)
            if hasattr(self, 'similarity_loss') and hasattr(self.similarity_loss, 'encoder'):
                encoder = self.similarity_loss.encoder
                # Try to get underlying HuggingFace model
                encoder_model = getattr(encoder, 'model', None)
                if encoder_model and hasattr(encoder_model, 'gradient_checkpointing_disable'):
                    logger.info_rank0("Disabling gradient checkpointing on similarity loss encoder")
                    encoder_model.gradient_checkpointing_disable()
                    if hasattr(encoder_model.config, 'gradient_checkpointing'):
                        encoder_model.config.gradient_checkpointing = False
        # Fallback to static graph mode for both models
        try:
            if hasattr(self.model, '_set_static_graph'):
                logger.info_rank0("Setting static graph for main model for distributed training")
                self.model._set_static_graph()
            elif hasattr(self.model, 'module') and hasattr(self.model.module, '_set_static_graph'):
                logger.info_rank0("Setting static graph on main model module for distributed training")
                self.model.module._set_static_graph()
            # Static graph for similarity loss encoder
            if hasattr(self, 'similarity_loss') and hasattr(self.similarity_loss, 'encoder'):
                encoder_model = getattr(self.similarity_loss.encoder, 'model', None)
                if encoder_model and hasattr(encoder_model, '_set_static_graph'):
                    logger.info_rank0("Setting static graph for similarity loss encoder for distributed training")
                    encoder_model._set_static_graph()
                elif encoder_model and hasattr(encoder_model, 'module') and hasattr(encoder_model.module, '_set_static_graph'):
                    logger.info_rank0("Setting static graph on similarity loss encoder module for distributed training")
                    encoder_model.module._set_static_graph()
        except Exception as e:
            logger.warning(f"Failed to set static graph: {str(e)}")
        # Ensure DDP's find_unused_parameters remains enabled
        if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'state'):
            logger.info_rank0("Checking DDP settings in accelerator")
            if hasattr(self.accelerator.state, 'ddp_plugin'):
                ddp_plugin = self.accelerator.state.ddp_plugin
                if hasattr(ddp_plugin, 'ddp_kwargs'):
                    ddp_plugin.ddp_kwargs['find_unused_parameters'] = True
        return optimizer

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        """Compute pairwise loss with optional CustomRewardModel support.
        
        If custom loss is enabled and the custom reward model is available,
        this will use the CustomRewardModel.
        Otherwise, it will fall back to the standard pairwise loss.
        """
        if self.use_custom_loss and self.custom_reward_model and self.similarity_loss:
            # Extract necessary information from inputs
            data = self._extract_similarity_loss_inputs(inputs)
            
            if data:
                query = data['query']
                original_doc = data['original_doc']
                rewritten_doc = data['rewritten_doc']
                top_k_docs = data['top_k_docs']
                
                # Calculate the similarity loss
                custom_loss = self.similarity_loss(query, original_doc, rewritten_doc, top_k_docs)
                
                # Ensure custom_loss is scalar
                if custom_loss.dim() > 0 and custom_loss.numel() > 1:
                    custom_loss = custom_loss.mean()
                
                # Get outputs from model for return value
                outputs = model(**inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    # For tuple outputs, the loss is typically the first element
                    model_loss = outputs[0] if outputs else torch.tensor(0.0, requires_grad=True)
                else:
                    # For outputs with loss attribute
                    model_loss = getattr(outputs, 'loss', torch.tensor(0.0, requires_grad=True))
                
                # Ensure model_loss is scalar too
                if model_loss.dim() > 0 and model_loss.numel() > 1:
                    model_loss = model_loss.mean()
                
                # Combine with model loss if needed or just use the custom loss
                combined_loss = custom_loss + model_loss * 0.5  # You can adjust the weight
                
                if return_outputs:
                    # Return combined loss and outputs
                    return combined_loss, outputs
                else:
                    return combined_loss
        
        # Fall back to standard loss computation
        outputs = model(**inputs)
        
        # Handle different output formats for standard loss computation too
        if return_outputs:
            if isinstance(outputs, tuple):
                return outputs[0], outputs
            else:
                return outputs.loss, outputs
        else:
            if isinstance(outputs, tuple):
                return outputs[0]
            else:
                return outputs.loss

    def _extract_similarity_loss_inputs(self, inputs):
        """
        Extract data needed for SimilarityLoss from model inputs.
        
        This method tries to parse the necessary information from standard
        reward model training data. For the ad_reward.json format, we extract:
        - Query from instruction or synthesize one
        - Original doc from the rejected text or input
        - Rewritten doc from the chosen text
        - Top-k docs from rankings
        
        Returns:
            Dict with query, original_doc, rewritten_doc, top_k_docs or None if failed
        """
        try:
            # Get tokenizer from model or trainer
            tokenizer = getattr(self, 'tokenizer', None)
            if tokenizer is None and hasattr(self, 'model'):
                tokenizer = getattr(self.model, 'tokenizer', None)
            
            # Try to extract input texts from input_ids if available
            input_texts = []
            if 'input_ids' in inputs and tokenizer is not None:
                input_texts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
            
            # Extract chosen/rejected texts for reward model format
            chosen_texts = []
            rejected_texts = []
            
            if 'chosen_input_ids' in inputs and tokenizer is not None:
                chosen_texts = tokenizer.batch_decode(inputs['chosen_input_ids'], skip_special_tokens=True)
            
            if 'rejected_input_ids' in inputs and tokenizer is not None:
                rejected_texts = tokenizer.batch_decode(inputs['rejected_input_ids'], skip_special_tokens=True)
            
            # If we've found valid texts, try to extract/construct query
            query = None
            if chosen_texts or rejected_texts or input_texts:
                # First try to extract query from input texts using regex pattern
                import re
                instruction_pattern = r"Human:\s*(.+?)\s*\n\nAssistant:"
                
                for text in input_texts:
                    match = re.search(instruction_pattern, text)
                    if match:
                        query = match.group(1).strip()
                        break
                
                # If no query found, try loading from ad_reward.json
                if not query:
                    try:
                        reward_file = 'data/ad_reward.json'
                        if os.path.exists(reward_file):
                            with open(reward_file, 'r', encoding='utf-8') as f:
                                reward_data = json.load(f)
                                if isinstance(reward_data, list) and len(reward_data) > 0:
                                    # Get the instruction from a random example
                                    example = random.choice(reward_data)
                                    instruction = example.get('instruction', '')
                                    
                                    # If instruction contains "Rewrite the advertisement", extract keywords
                                    if 'Rewrite the advertisement' in instruction:
                                        text = example.get('input', '')
                                        # Extract keywords for synthetic query
                                        words = text.lower().split()
                                        keywords = [w for w in words if len(w) > 5 and w.isalpha()]
                                        if keywords:
                                            num_words = min(len(keywords), random.randint(2, 3))
                                            selected = random.sample(keywords, num_words)
                                            query = ' '.join(selected)
                                        else:
                                            query = "product information"
                                    else:
                                        # Extract query part from instruction if possible
                                        query = instruction
                    except Exception as e:
                        print(f"Error loading ad_reward.json: {e}")
                
                # If still no query, fall back to a default
                if not query:
                    query = "What products do you recommend?"
                
                # Determine original and rewritten docs
                original_doc = None
                rewritten_doc = None
                
                # For reward model format: rejected = original, chosen = rewritten
                if rejected_texts and chosen_texts:
                    original_doc = rejected_texts[0]
                    rewritten_doc = chosen_texts[0]
                # If only chosen available, use input as original
                elif chosen_texts and input_texts:
                    original_doc = input_texts[0]
                    rewritten_doc = chosen_texts[0]
                # If not enough data, try to use dataset if available
                elif hasattr(self, 'train_dataset') and self.train_dataset:
                    # Get a random example from the dataset
                    index = random.randint(0, len(self.train_dataset) - 1)
                    example = self.train_dataset[index]
                    
                    # Extract original and rewritten docs
                    original_doc = example.get('input', example.get('rejected', ''))
                    rewritten_doc = example.get('chosen', '')
                else:
                    # Not enough data to proceed
                    return None
                
                # Get top-k docs from rankings
                rankings = load_rankings_file()
                ads_by_id = load_ads_by_id()
                top_k_docs = build_top_k_docs(query, rankings, ads_by_id)
                
                # If no top-k docs found, create variations of the original
                if not top_k_docs:
                    top_k_docs = []
                    
                    # Use classified ads to find similar domain/subdomain docs
                    classified_ads = load_classified_ads()
                    query_responses = load_query_responses()
                    
                    # Try to identify domain/subdomain of the original doc
                    domain = subdomain = None
                    
                    # Look through query responses to guess domain/subdomain
                    for q, info in query_responses.items():
                        if original_doc and original_doc in info.get('retrieved_context', ''):
                            domain = info.get('domain')
                            subdomain = info.get('subdomain')
                            break
                    
                    # If domain/subdomain found, look for other ads in same category
                    if domain and subdomain:
                        ads_by_id = load_ads_by_id()
                        for ad_id, ad_info in classified_ads.items():
                            if ad_info.get('domain') == domain and ad_info.get('subdomain') == subdomain:
                                if ad_id in ads_by_id:
                                    ad = ads_by_id[ad_id]
                                    if 'text' in ad and ad['text'] != original_doc:
                                        top_k_docs.append(ad['text'])
                                        if len(top_k_docs) >= 3:
                                            break
                    
                    # If still no docs, create variations
                    if not top_k_docs and original_doc:
                        # Add a variation with sentence shuffling
                        sentences = original_doc.split('.')
                        if len(sentences) > 2:
                            random.shuffle(sentences)
                            top_k_docs.append('. '.join(sentences[:len(sentences)//2]) + '.')
                        
                        # Add a variation with word removal
                        words = original_doc.split()
                        if len(words) > 20:
                            removals = random.sample(range(len(words)), min(10, len(words) // 3))
                            variation = ' '.join([w for i, w in enumerate(words) if i not in removals])
                            top_k_docs.append(variation)
                
                # Make sure the original document is not in top_k_docs
                top_k_docs = [doc for doc in top_k_docs if doc != original_doc]
                
                # If we still don't have any top-k docs, add a simple variation
                if not top_k_docs and original_doc:
                    top_k_docs.append(original_doc[:len(original_doc)//2] + "...")
                
                return {
                    'query': query,
                    'original_doc': original_doc,
                    'rewritten_doc': rewritten_doc,
                    'top_k_docs': top_k_docs
                }
            
            return None
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error extracting similarity loss inputs: {str(e)}")
            return None

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")
        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: list[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))

            writer.write("\n".join(res))
