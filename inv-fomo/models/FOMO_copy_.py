 # ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection
# Orr Zohar, Jackson Wang, Serena Yeung
# ------------------------------------------------------------------------
# Modified from Transformers: 
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/owlvit/modeling_owlvit.py
# -----------------------------------------------------------------------
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection, OwlViTConfig, OwlViTModel
from transformers.models.owlvit.modeling_owlvit import *
from torch.special import digamma, gammaln
from .utils import *
from .few_shot_dataset import FewShotDataset, aug_pipeline, collate_fn

from util import box_ops
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
from scipy.stats import gaussian_kde
import torch
from scipy import stats
import math
from PIL import Image

def split_into_chunks(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]



class UnkDetHead(nn.Module):
    def __init__(self, method, known_dims, att_W, **kwargs):
        super(UnkDetHead, self).__init__()
        print("UnkDetHead", method)
        self.method = method
        self.known_dims = known_dims
        self.att_W = att_W
        self.process_mcm = nn.Softmax(dim=-1)

        if "sigmoid" in method:
            self.process_logits = nn.Sigmoid()
            self.proc_obj = True
        elif "softmax" in method:
            self.process_logits = nn.Softmax(dim=-1)
            self.proc_obj = True
        else:
            self.proc_obj = False

    def forward(self, att_logits):
        logits = att_logits @ self.att_W
        k_logits = logits[..., :self.known_dims]
        unk_logits = logits[..., self.known_dims:].max(dim=-1, keepdim=True)[0]
        logits = torch.cat([k_logits, unk_logits], dim=-1)
        objectness = torch.ones_like(unk_logits).squeeze(-1)

        if "mean" in self.method:
            sm_logits = self.process_logits(att_logits)
            objectness = sm_logits.mean(dim=-1, keepdim=True)[0]

        elif "max" in self.method:
            sm_logits = self.process_logits(att_logits)
            objectness = sm_logits.max(dim=-1, keepdim=True)[0]

        if "mcm" in self.method:
            mcm = self.process_mcm(k_logits).max(dim=-1, keepdim=True)[0]
            objectness *= (1 - mcm)

        if self.proc_obj:
            objectness -= objectness.mean()
            objectness /= objectness.std()
            objectness = torch.sigmoid(objectness)

        return logits, objectness.squeeze(-1)




class OwlViTTextTransformer(OwlViTTextTransformer):
    @add_start_docstrings_to_model_forward(OWLVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=OwlViTTextConfig)
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        num_samples, seq_len = input_shape  # num_samples = batch_size * num_max_text_queries
        # OWLVIT's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(num_samples, seq_len).to(hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [num_samples, seq_len] -> [num_samples, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # take features from the end of tokens embedding (end of token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(torch.int).argmax(dim=-1).to(last_hidden_state.device),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len)
        mask.fill_(torch.tensor(float("-inf")))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask


@add_start_docstrings(OWLVIT_START_DOCSTRING)
class OurOwlViTModel(OwlViTModel):
    @add_start_docstrings_to_model_forward(OWLVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OwlViTOutput, config_class=OwlViTConfig)
    def forward_vision(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
        """
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get embeddings for all text queries in all batch samples

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        # normalized features
        image_embeds = image_embeds / torch.linalg.norm(image_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        return image_embeds, vision_outputs

    def forward_text(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
            """

        # Get embeddings for all text queries in all batch samples
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        text_embeds_norm = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        return text_embeds_norm, text_outputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_loss: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_base_image_embeds: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, OwlViTOutput]:
        r"""
        Returns:
            """
        # Use OWL-ViT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # normalized features
        image_embeds, vision_outputs = self.forward_vision(pixel_values=pixel_values,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)

        text_embeds_norm, text_outputs = self.forward_text(input_ids=input_ids, attention_mask=attention_mask,
                                                           output_attentions=output_attentions,
                                                           output_hidden_states=output_hidden_states,
                                                           return_dict=return_dict)

        # cosine similarity as logits and set it on the correct device
        logit_scale = self.logit_scale.exp().to(image_embeds.device)

        logits_per_text = torch.matmul(text_embeds_norm, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = owlvit_loss(logits_per_text)

        if return_base_image_embeds:
            warnings.warn(
                "`return_base_image_embeds` is deprecated and will be removed in v4.27 of Transformers, one can"
                " obtain the base (unprojected) image embeddings from outputs.vision_model_output.",
                FutureWarning,
            )
            last_hidden_state = vision_outputs[0]
            image_embeds = self.vision_model.post_layernorm(last_hidden_state)
        else:
            text_embeds = text_embeds_norm

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return OwlViTOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class FOMO(nn.Module):
    """This is the OWL-ViT model that performs open-vocabulary object detection"""

    def __init__(self, args, model_name, known_class_names, unknown_class_names, templates, image_conditioned, device):
        """ Initializes the model.
        Parameters:
            model_name: the name of the huggingface model to use
            known_class_names: list of the known class names
            templates:
            attributes: dict of class names (keys) and the corresponding attributes (values).

        """
        super().__init__()
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
        self.model.owlvit = OurOwlViTModel.from_pretrained(model_name).to(device)
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)

        self.known_class_names = known_class_names
        self.unknown_class_names = unknown_class_names
        all_classnames = known_class_names + unknown_class_names
        self.all_classnames = all_classnames
        self.templates = templates
        self.num_attributes_per_class = args.num_att_per_class

        if image_conditioned:
            fs_dataset = FewShotDataset(
                args.dataset,
                args.image_conditioned_file,
                args.image_inv_file,
                self.known_class_names,
                args.num_few_shot,
                self.processor,
                args.data_task,
                aug_pipeline,

            )

            fs_dataloader = DataLoader(dataset=fs_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       collate_fn=collate_fn,
                                       shuffle=True,
                                       drop_last=True)

            if args.use_attributes:
                with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.attributes_file}', 'r') as f:
                    attributes = json.loads(f.read())

                self.attributes_texts = [f"object which (is/has/etc) {cat} is {a}" for cat, att in attributes.items()
                                         for a in att]
                
                # with open("/root/autodl-tmp/inv-fomo/watt/surgical_att.json", "w", encoding="utf-8") as f:
                #     json.dump(self.attributes_texts, f, ensure_ascii=False, indent=4)
                    
                self.att_W = torch.rand(len(self.attributes_texts), len(known_class_names), device=device)
                img_name = "/root/autodl-tmp/inv-fomo/data/RWD/JPEGImages/Aquatic/IMG_2578_jpeg_jpg.rf.fa977ca797fc5746af6bc5991a03c80a.jpg"
                # img_name = "/root/autodl-tmp/inv-fomo/data/RWD/JPEGImages/Game/picture_94_background_0_png_jpg.rf.213f880bff7cff64e772a6bc98d20904.jpg"
                boxes = [
                            [0.0, 0.0439453125, 0.1796875, 0.1630859375],
                            [0.2109375, 0.1240234375, 0.38020833333333337, 0.224609375],
                            [0.0234375, 0.1005859375, 0.20963541666666666, 0.177734375],
                            [0.6901041666666666, 0.1953125, 0.9205729166666666, 0.2646484375],
                            [0.38671875, 0.162109375, 0.578125, 0.2255859375],
                            [0.18619791666666666, 0.193359375, 0.40234375, 0.25390625],
                            [0.09895833333333333, 0.0, 0.2682291666666667, 0.0595703125],
                            [0.68359375, 0.1259765625, 0.8606770833333334, 0.1796875],
                            [0.4869791666666667, 0.1982421875, 0.6484375, 0.2470703125],
                            [0.77734375, 0.1513671875, 0.9270833333333334, 0.1962890625],
                            [0.3736979166666667, 0.0, 0.46875, 0.0439453125]
                        ]
                # boxes=[          
                #           [
                #             0.001953125,
                #             0.33837890625,
                #             0.8372395833333334,
                #             0.66943359375
                #           ]
                                        # ]
                self.get_obj_embeddings(img_name, boxes)

                with torch.no_grad():
                    mean_known_query_embeds,  inv_mean_known_query_embeds, embeds_dataset,inv_embeds_dataset = self.get_mean_embeddings(fs_dataset)
                    text_mean_norm, att_query_mask = self.prompt_template_ensembling(self.attributes_texts, templates)
                    self.att_embeds = text_mean_norm.detach().clone().to(device)
                    self.att_query_mask = att_query_mask.to(device)

                if args.att_selection:

                    self.attribute_selection(embeds_dataset, inv_embeds_dataset, args.neg_sup_ep * 500, args.neg_sup_lr,mean_known_query_embeds)
                    # self.attribute_selection(embeds_dataset, args.neg_sup_ep * 500, args.neg_sup_lr)
                    # selected_idx = torch.where(torch.sum(self.att_W, dim=1) != 0)[0]
                    # self.att_embeds = torch.index_select(self.att_embeds, 1, selected_idx)
                    # self.att_W = torch.index_select(self.att_W, 0, selected_idx)
                    # print(f"Selected {len(selected_idx.tolist())} attributes from {len(self.attributes_texts)}")
                    # self.attributes_texts = [self.attributes_texts[i] for i in selected_idx.tolist()]

                # self.att_W = F.normalize(self.att_W, p=1, dim=0).to(device)
                # self.att_query_mask = None

                # if args.att_adapt:
                #     self.adapt_att_embeddings(mean_known_query_embeds)

                if args.att_refinement:
                    self.attribute_refinement(fs_dataloader, args.neg_sup_ep, args.neg_sup_lr)

                if args.use_attributes:
                    self.att_embeds = torch.cat([self.att_embeds,
                                                 torch.matmul(self.att_embeds.squeeze().T, self.att_W).mean(1,
                                                                                                            keepdim=True).T.unsqueeze(
                                                     0)], dim=1)
                else:
                    with torch.no_grad():
                        unknown_query_embeds, _ = self.prompt_template_ensembling(unknown_class_names, templates)
                    self.att_embeds = torch.cat([self.att_embeds, unknown_query_embeds], dim=1)

                eye_unknown = torch.eye(1, device=self.device)
                self.att_W = torch.block_diag(self.att_W, eye_unknown)
            else:
                ## run simple baseline
                with torch.no_grad():
                    mean_known_query_embeds, _, _,_ = self.get_mean_embeddings(fs_dataset)
                    unknown_query_embeds, _ = self.prompt_template_ensembling(unknown_class_names, templates)
                    self.att_embeds = torch.cat([mean_known_query_embeds, unknown_query_embeds], dim=1)

                self.att_W = torch.eye(len(known_class_names) + 1, device=device)
                self.att_query_mask = None
        else:
            self.att_embeds, self.att_query_mask = self.prompt_template_ensembling(all_classnames, templates)
            self.att_W = torch.eye(len(all_classnames), device=self.device)

        self.unk_head = UnkDetHead(args.unk_method, known_dims=len(known_class_names),
                                   att_W=self.att_W, device=device)


        
    def get_obj_embeddings(self, image_name, boxes):
        with torch.no_grad():
            img = Image.open(image_name).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.model.device)
            query_feature_map = self.model.image_embedder(pixel_values=inputs['pixel_values'])[0]
            batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
            query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
            # Get top class embedding and best box index for each query image in batch
            select_list = []
            obj_list = []
            query_list = []
            for i in range(len(boxes)):
                current_box = [torch.tensor(boxes[i], device=self.model.device).unsqueeze(0)]
                query_embeds, selected, _, missing_indexes, obj_idx,class_embeds = self.embed_image_query(query_image_feats, query_feature_map, current_box)
                select_list.append(selected)
                obj_list.append(obj_idx)
                query_list.append(class_embeds)


        torch.save(query_list[0], "/root/autodl-tmp/inv-fomo/watt/aquatic_class_embeds.pt")
        torch.save(select_list, '/root/autodl-tmp/inv-fomo/watt/aquatic_select.pt')
        torch.save(obj_list, '/root/autodl-tmp/inv-fomo/watt/aquatic_obj.pt')
        import ipdb;ipdb.set_trace()
        return
        
    

    def attribute_refinement(self, fs_dataloader, epochs, lr):
        optimizer = torch.optim.AdamW([self.att_embeds], lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.att_embeds.requires_grad_()
        # Create tqdm object for displaying progress
        pbar = tqdm(range(epochs), desc="Refining selected attributes:")
        for _ in pbar:
            mean_loss = []
            for batch_idx, batch in enumerate(fs_dataloader):
                optimizer.zero_grad()
                with torch.no_grad():
                    image_embeds, targets = self.image_guided_forward(batch["image"].to(self.device),
                                                                      bboxes=batch["bbox"],
                                                                      cls=batch["label"])
                    import ipdb;ipdb.set_trace()
                    

                    if image_embeds is None:
                        continue
                    targets = torch.stack(targets).to(self.device)

                cos_sim = cosine_similarity(image_embeds, self.att_embeds, dim=-1)
                logits = torch.matmul(cos_sim, self.att_W)
                loss = criterion(logits, targets)  # Compute loss
                loss.backward()
                optimizer.step()  # Update cls_embeds using gradients
                mean_loss.append(loss.detach().cpu().numpy())

            # Update progress bar with current mean loss
            pbar.set_postfix({"loss": np.mean(mean_loss)}, refresh=True)
        self.att_embeds.requires_grad_(False)
        return


    def get_class_embeddings_and_tables(self, fs_dataloader):
        target_embeddings = []
        class_ids = []
        for class_id, embeddings_batches in fs_dataloader.items():
            for batch in embeddings_batches:
                target_embeddings.append(batch)
                class_ids.extend([class_id] * batch.shape[0])

        # Concatenate target embeddings
        image_embeddings = torch.cat(target_embeddings, dim=0).to(self.device)

        # Create one-hot encoded targets
        num_classes = len(fs_dataloader)
        # targets = F.one_hot(torch.tensor(class_ids), num_classes=num_classes).float().to(self.device)
        targets = torch.tensor(class_ids, dtype=torch.long).to(self.device) # CE loss

        return image_embeddings, targets

    # def compute_invariance_score(self, cos_sim, inv_cos_sim, eps=1e-6):
    #
    #     delta = cos_sim - inv_cos_sim
    #     var = torch.var(delta, dim=0)
    #     inv_score = torch.exp(-var)
    #
    #     inv_score = torch.clamp(inv_score, min=eps, max=1.0)
    #
    #     return inv_score


    # def attribute_selection(self, fs_dataloader, epochs, lr):
    #     target_embeddings = []
    #     class_ids = []
    #     for class_id, embeddings_batches in fs_dataloader.items():
    #         for batch in embeddings_batches:
    #             target_embeddings.append(batch)
    #             class_ids.extend([class_id] * batch.shape[0])
    #
    #     # Concatenate target embeddings
    #     image_embeddings = torch.cat(target_embeddings, dim=0).to(self.device)
    #
    #     # Create one-hot encoded targets
    #     num_classes = len(fs_dataloader)
    #     targets = F.one_hot(torch.tensor(class_ids), num_classes=num_classes).float().to(self.device)
    #
    #     optimizer = torch.optim.AdamW([self.att_W], lr=lr)
    #     criterion = nn.BCEWithLogitsLoss()
    #     self.att_W.requires_grad_()
    #     lambda1 = 0.01
    #
    #     # Create tqdm object for displaying progress
    #     pbar = tqdm(range(epochs), desc="Attribute selection:")
    #     for _ in pbar:
    #         optimizer.zero_grad()
    #         self.att_W.data = torch.clamp(self.att_W.data, 0, 1)
    #         cos_sim = cosine_similarity(image_embeddings, self.att_embeds, dim=-1)
    #         logits = torch.matmul(cos_sim, self.att_W)
    #
    #         loss = criterion(logits, targets)  # Compute loss
    #         l1_reg = torch.norm(self.att_W, p=1)
    #         loss += lambda1 * l1_reg
    #         loss.backward()
    #         optimizer.step()  # Update cls_embeds using gradients
    #         pbar.set_postfix({"loss": loss}, refresh=True)
    #
    #     with torch.no_grad():
    #         _, top_indices = torch.topk(self.att_W.view(-1), num_classes * self.num_attributes_per_class)
    #         self.att_W.fill_(0)  # Reset all attributes to 0
    #         self.att_W.view(-1)[top_indices] = 1
    #
    #     self.att_W.requires_grad_(False)
    #     return

    def memory_efficient_cos_sim(self, x1, x2):
        x1_ = x1.squeeze(1)
        x2_ = x2.squeeze(0)
        x1_norm = F.normalize(x1_, p=2, dim=-1)
        x2_norm = F.normalize(x2_, p=2, dim=-1)

        return torch.mm(x1_norm, x2_norm.t()) 



    
    def attribute_selection(self, fs_dataloader, inv_fs_dataloader, epochs, lr, mean_known_query_embeds):

        image_embeddings, targets = self.get_class_embeddings_and_tables(fs_dataloader)
        num_classes = len(fs_dataloader)

        inv_image_embeddings, _ = self.get_class_embeddings_and_tables(inv_fs_dataloader)
        # torch.save(inv_image_embeddings, "/root/autodl-tmp/inv-fomo/watt/aquatic_known_u.pt")
        # import ipdb;ipdb.set_trace()

        lr_w = lr
        lr_embed2 = 1e-3


        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()
        criterion1 = nn.MSELoss()
        lambda1 = 1 # t1:Aquntic:1.85, Medicl:0.78,Sugical:0.27,Aerial:2.1,Game:0.68; t2:Aquntic:1.85, Medicl:0.78,Sugical:0.27,Aerial:2.5,Game：0.5

        self.att_W.requires_grad_(True)


        optimizer = torch.optim.AdamW([
            {"params": [self.att_W], "lr": lr_w}


        ])

        epochs_ = 1000

        total_epochs = epochs + epochs_

        pbar = tqdm(range(total_epochs), desc="Attribute selection:")

        for epoch in pbar:

            optimizer.zero_grad()

            if epoch == epochs:
                with torch.no_grad():
                    # cos_sim = cosine_similarity(image_embeddings, self.att_embeds, dim=-1)
                    # inv_cos_sim = cosine_similarity(inv_image_embeddings, self.att_embeds, dim=-1)
                    cos_sim = self.memory_efficient_cos_sim(image_embeddings, self.att_embeds)
                    inv_cos_sim = self.memory_efficient_cos_sim(inv_image_embeddings, self.att_embeds)

                    inv_score = self.compute_invariance_score(cos_sim, inv_cos_sim, targets, num_classes)
                    
                    # torch.save(self.att_W, "/root/autodl-tmp/inv-fomo/watt/aerial_att.pt")
                    # torch.save(inv_score.T, "/root/autodl-tmp/inv-fomo/watt/aerial_inv.pt")
                    # import ipdb;ipdb.set_trace()
                    # final_score = self.att_W * inv_score.T
                    final_score = self.att_W


                    _, top_indices = torch.topk(
                        final_score.view(-1),
                        num_classes * self.num_attributes_per_class
                    )

                    self.att_W.fill_(0)
                    self.att_W.view(-1)[top_indices] = 1
                    self.att_W.requires_grad_(False)
                    self.att_embeds.requires_grad_(False)

                    selected_idx = torch.where(torch.sum(self.att_W, dim=1) != 0)[0]
                    # torch.save(self.att_W, "/root/autodl-tmp/inv-fomo/watt/surgical_idx.pt")
                    self.att_embeds = torch.index_select(
                        self.att_embeds, 1, selected_idx
                    ).detach().requires_grad_(True)
                    # torch.save(self.att_embeds, "/root/autodl-tmp/inv-fomo/watt/aquatic_oringin_embeds.pt")
                    self.att_W = torch.index_select(self.att_W, 0, selected_idx)

                print(f"Selected {len(selected_idx.tolist())} attributes from {len(self.attributes_texts)}")
                self.att_W = F.normalize(self.att_W, p=1, dim=0).to(self.device)
                self.att_query_mask = None


                self.att_W.requires_grad_(False)
                self.att_embeds.requires_grad_(True)

                optimizer = torch.optim.AdamW([
                    {"params": [self.att_embeds], "lr": lr_embed2}
                ])

                continue

            self.att_W.data = torch.clamp(self.att_W.data, 0, 1)
            # cos_sim = cosine_similarity(image_embeddings, self.att_embeds, dim=-1)
            cos_sim = self.memory_efficient_cos_sim(image_embeddings, self.att_embeds)

            logits = torch.matmul(cos_sim, self.att_W)

            if epoch < epochs:
                
                # inv_cos_sim = cosine_similarity(inv_image_embeddings, self.att_embeds, dim=-1)
                inv_cos_sim = self.memory_efficient_cos_sim(inv_image_embeddings, self.att_embeds)
                T = 0.07
                
                inv_logits = torch.matmul(inv_cos_sim, self.att_W)
                inv_loss = F.kl_div(F.log_softmax(inv_logits/ T,dim=1), F.log_softmax(logits / T,dim=1), reduction='batchmean', log_target=True)
                
                class_loss = criterion(logits, targets)
                loss = class_loss + lambda1 * inv_loss 
                # import ipdb;ipdb.set_trace()
                print(f"inv_loss:{inv_loss}, ce_loss:{criterion(logits, targets)}")

            else:


                output = torch.matmul(self.att_W.T.unsqueeze(0), self.att_embeds)
                loss = criterion1(output, mean_known_query_embeds)



            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()}, refresh=True)

        self.att_W.requires_grad_(False)
        self.att_embeds.requires_grad_(False)
        # torch.save(self.att_embeds, "/root/autodl-tmp/inv-fomo/watt/aquatic_embeds.pt")

        return




    # def attribute_selection(self, fs_dataloader, inv_fs_dataloader, epochs, lr):
    #     # 1. 初始化数据与参数
    #     img_embeds, targets = self.get_class_embeddings_and_tables(fs_dataloader)
    #     inv_img_embeds, _ = self.get_class_embeddings_and_tables(inv_fs_dataloader)
    #
    #     num_classes = len(fs_dataloader)
    #     total_attrs = self.att_embeds.shape[1]
    #     target_attrs = num_classes * self.num_attributes_per_class
    #
    #     # 初始掩码（1表示保留，0表示剪枝）
    #     mask = torch.ones(total_attrs, device=self.att_embeds.device)
    #
    #     # 优化器
    #     self.att_W.requires_grad_(True)
    #     self.att_embeds.requires_grad_(True)
    #     optimizer = torch.optim.AdamW([
    #         {"params": [self.att_W], "lr": lr},
    #         {"params": [self.att_embeds], "lr": 1e-3}
    #     ], weight_decay=1e-5)
    #
    #     pbar = tqdm(range(epochs), desc="Refining Attributes")
    #     lambda_inv = 35.0
    #     lambda_sparse = 0.01
    #     for epoch in pbar:
    #         optimizer.zero_grad()
    #
    #         # --- A. 计算当前阶段应保留的数量 (余弦退火调度) ---
    #         # 这种调度让前期剪枝快，后期剪枝慢，给模型留出微调时间
    #         progress = epoch / epochs
    #         current_keep_ratio = target_attrs / total_attrs + \
    #                              0.5 * (1 - target_attrs / total_attrs) * (1 + math.cos(math.pi * progress))
    #         current_keep_num = max(target_attrs, int(total_attrs * current_keep_ratio))
    #
    #         # --- B. 前向传播与损失计算 ---
    #         # 计算相似度矩阵 (N_img, N_attr)
    #         cos_sim = cosine_similarity(img_embeds, self.att_embeds, dim=-1)
    #
    #         # 应用掩码：确保被剪枝的属性不参与计算
    #         masked_cos_sim = cos_sim * mask.view(1, -1)
    #         logits = torch.matmul(masked_cos_sim, self.att_W)
    #
    #         # 判别性损失 (CE) + 不变性损失 (KL)
    #         class_loss = F.cross_entropy(logits, targets)
    #
    #         inv_cos_sim = F.cosine_similarity(inv_img_embeds, self.att_embeds, dim=-1)
    #         inv_loss = F.kl_div(
    #             F.log_softmax(cos_sim, dim=1),
    #             F.log_softmax(inv_cos_sim, dim=1),
    #             reduction='batchmean', log_target=True
    #         )
    #
    #         # 稀疏性惩罚 (引导 att_W 趋向于 0)
    #         sparse_loss = torch.norm(self.att_W, p=1)
    #
    #         loss = class_loss + lambda_inv * inv_loss * self.att_embeds.shape[-1]
    #         loss.backward()
    #         optimizer.step()
    #
    #         # --- C. 执行课程裁剪 (定期更新掩码) ---
    #         # 我们不物理删除，只把掩码设为0
    #         if epoch % 5 == 0 and epoch < epochs * 0.8:  # 在前80%的训练时间内完成剪枝
    #             with torch.no_grad():
    #                 inv_score = self.compute_invariance_score(cos_sim, inv_cos_sim)
    #                 # 综合得分：属性在所有类上的最大权重绝对值 * 不变性分
    #                 importance = torch.max(torch.abs(self.att_W), dim=1)[0] * inv_score
    #
    #                 # 更新掩码
    #                 _, top_idx = torch.topk(importance, current_keep_num)
    #                 new_mask = torch.zeros_like(mask)
    #                 new_mask[top_idx] = 1
    #                 mask = new_mask
    #
    #                 # 立即置零已剪枝权重及其梯度
    #                 self.att_W.data *= mask.unsqueeze(1)
    #                 if self.att_W.grad is not None:
    #                     self.att_W.grad *= mask.unsqueeze(1)
    #
    #         pbar.set_postfix({
    #             "L": f"{loss.item():.3f}",
    #             "Keep": current_keep_num,
    #             "Sparse": f"{sparse_loss.item():.1f}"
    #         })
    #     # 4. 训练结束：物理裁剪
    #     self._compact_parameters(mask)
    #     self.att_embeds.requires_grad_(False)
    #     self.att_W.requires_grad_(False)
    #     return

    def _compact_parameters(self, mask):
        """将 Mask 为 1 的参数提取出来，实现物理上的降维"""
        with torch.no_grad():
            keep_idx = torch.where(mask == 1)[0]
            # 物理收缩
            self.att_embeds = self.att_embeds[:, keep_idx, :].clone()
            self.att_W = self.att_W[keep_idx, :].clone()

            # 重新封装为 Parameter
            # self.att_embeds = nn.Parameter(new_embeds)
            # self.att_W = nn.Parameter(new_W)
        print(f"Final reduction complete. Remaining attributes: {self.att_W.shape[0]}")

    # def compute_invariance_score(self, cos_sim, inv_cos_sim):
    #     """计算不变性分数：差值方差越小，分数越高"""
    #     # delta shape: (batch, num_attrs)
    #     delta = cos_sim - inv_cos_sim
    #     var = torch.var(delta, dim=0)
    #     # 使用 exp(-var) 将方差映射到 (0, 1]，方差大(不稳定)则分数趋近0
    #     inv_score = torch.exp(-var)
    #     return torch.clamp(inv_score, min=1e-6, max=1.0)
    
    def compute_invariance_score(self, cos_sim, inv_cos_sim, targets, num_classes):

        T = 0.07  
        inv_score = torch.zeros((num_classes, self.att_embeds.shape[1]), device=self.device)

        for cla in range(num_classes):

            idx = (targets == cla)
            if not idx.any():
                continue
                
            cos_c = cos_sim[idx]
            inv_cos_c = inv_cos_sim[idx]

            p = F.log_softmax(cos_c / T, dim=-1)
            q = F.log_softmax(inv_cos_c / T, dim=-1)

            kl_elementwise = F.kl_div(q, p, reduction='none', log_target=True)

            kl_per_attribute = kl_elementwise.mean(dim=0)
            
            max_kl = kl_per_attribute.max()
            min_kl = kl_per_attribute.min()

            norm_kl = (kl_per_attribute - min_kl) / (max_kl - min_kl + 1e-8)
            inv_score[cla, :] = torch.exp(-norm_kl)

        return torch.clamp(inv_score, 1e-6, 1.0)
    
    
    def get_mean_embeddings(self, fs_dataset):
        dataset = {i: [] for i in range(len(self.known_class_names))}
        inv_dataset = {i: [] for i in range(len(self.known_class_names))}  # For inv_image_batch

        for img_batch in split_into_chunks(range(len(fs_dataset)), 3):
            image_batch = collate_fn([fs_dataset.get_no_aug(i) for i in img_batch])
            inv_image_batch = collate_fn([fs_dataset.get_cit_aug(i) for i in img_batch])

            grouped_data = defaultdict(list)
            inv_grouped_data = defaultdict(list)

            # Grouping image data
            for bbox, label, image in zip(image_batch['bbox'], image_batch['label'], image_batch['image']):
                grouped_data[label].append({'bbox': bbox, 'image': image})

            # Grouping inv_image_batch data
            for bbox, label, image in zip(inv_image_batch['bbox'], inv_image_batch['label'], inv_image_batch['image']):
                inv_grouped_data[label].append({'bbox': bbox, 'image': image})

            # Processing image data
            for l, data in grouped_data.items():
                tmp = self.image_guided_forward(torch.stack([d["image"] for d in data]).to(self.device),
                                                [d["bbox"] for d in data]).cpu()
                dataset[l].append(tmp)

            # Processing inv_image_batch data
            for l, data in inv_grouped_data.items():
                tmp = self.image_guided_forward(torch.stack([d["image"] for d in data]).to(self.device),
                                                [d["bbox"] for d in data]).cpu()

                inv_dataset[l].append(tmp)

        # Return both dataset and inv_dataset
        mean_embeddings = torch.cat([torch.cat(dataset[i], 0).mean(0) for i in range(len(self.known_class_names))],
                                    0).unsqueeze(0).to(self.device)
        inv_mean_embeddings = torch.cat(
            [torch.cat(inv_dataset[i], 0).mean(0) for i in range(len(self.known_class_names))], 0).unsqueeze(0).to(
            self.device)

        return mean_embeddings, inv_mean_embeddings, dataset, inv_dataset

    def adapt_att_embeddings(self, mean_known_query_embeds):
        self.att_embeds.requires_grad_()  # Enable gradient computation
        optimizer = torch.optim.AdamW([self.att_embeds], lr=1e-3)  # Define optimizer
        criterion = torch.nn.MSELoss()  # Define loss function

        for i in range(1000):
            optimizer.zero_grad()  # Clear gradients

            output = torch.matmul(self.att_W.T.unsqueeze(0), self.att_embeds)
            loss = criterion(output, mean_known_query_embeds)  # Compute loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update cls_embeds using gradients

            if i % 100 == 0:
                print(f"Step {i}, Loss: {loss.item()}")

        self.att_embeds.requires_grad_(False)

    def prompt_template_ensembling(self, classnames, templates):
        print('performing prompt ensembling')
        text_sum = torch.zeros((1, len(classnames), self.model.owlvit.text_embed_dim)).to(self.device)

        for template in templates:
            print('Adding template:', template)
            # Generate text for each class using the template
            class_texts = [template.replace('{c}', classname.replace('_', ' ')) for classname in
                           classnames]

            text_tokens = self.processor(text=class_texts, return_tensors="pt", padding=True, truncation=True).to(
                self.device)

            # Forward pass through the text encoder
            text_tensor, query_mask = self.forward_text(**text_tokens)

            text_sum += text_tensor

        # Calculate mean of text embeddings
        # text_mean = text_sum / text_count
        text_norm = text_sum / torch.linalg.norm(text_sum, ord=2, dim=-1, keepdim=True) + 1e-6
        return text_norm, query_mask

    def embed_image_query(
            self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor,
            each_query_boxes
    ) -> torch.FloatTensor:
        _, class_embeds = self.model.class_predictor(query_image_features)
        pred_boxes = self.model.box_predictor(query_image_features, query_feature_map)
        pred_boxes_as_corners = box_ops.box_cxcywh_to_xyxy(pred_boxes)

        # Loop over query images
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device
        bad_indexes = []
        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor(each_query_boxes[i], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            # If there are no overlapping boxes, fall back to generalized IoU
            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            # Use an adaptive threshold to include all boxes within 80% of the best IoU
            iou_threshold = torch.max(ious) * 0.8

            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]

                # --- 优化版：纯净背景排斥逻辑 ---
                # 1. 构建纯净背景掩码 (排除掉所有与目标有交集、IoU较高的框)
                num_total_boxes = class_embeds.shape[1]
                bg_mask = torch.ones(num_total_boxes, dtype=torch.bool, device=pred_boxes_device)

                # 为了更严谨，我们可以把 IoU 大于 0 的框都从背景池中剔除，确保背景纯净
                # 或者直接剔除 selected_inds。这里选择剔除 selected_inds
                bg_mask[selected_inds.squeeze(1)] = False

                # 2. 计算纯净背景的平均特征
                pure_bg_embeds = class_embeds[i][bg_mask]
                pure_bg_mean = torch.mean(pure_bg_embeds, dim=0)

                # 3. 计算候选特征与纯净背景的余弦相似度 (使用 F.cosine_similarity 排除了模长干扰)
                # 将 pure_bg_mean 扩展为 (1, D) 以便进行广播计算
                sim_to_bg = F.cosine_similarity(selected_embeddings, pure_bg_mean.unsqueeze(0), dim=-1)

                # 4. 寻找与背景相似度最低（排斥最强）的框
                best_local_ind = torch.argmin(sim_to_bg)
                best_box_ind = selected_inds[best_local_ind]


#                 mean_embeds = torch.mean(class_embeds[i], axis=0)
#                 mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
#                 best_box_ind = selected_inds[torch.argmin(mean_sim)]

                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)


            else:
                bad_indexes.append(i)

        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)

        else:
            query_embeds, box_indices = None, None
        # return query_embeds, box_indices, pred_boxes, bad_indexes
        return query_embeds, box_indices, pred_boxes, bad_indexes, selected_inds,class_embeds

    def image_guided_forward(
            self,
            query_pixel_values: Optional[torch.FloatTensor] = None, bboxes=None, cls=None
    ):
        # Compute feature maps for the input and query images
        # save_tensor_as_image_with_bbox(query_pixel_values[0].cpu(), bboxes[0][0], f'tmp/viz/{cls}_img.png')
        import ipdb;ipdb.set_trace()
        query_feature_map = self.model.image_embedder(pixel_values=query_pixel_values)[0]
        batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
        query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
        # Get top class embedding and best box index for each query image in batch
        query_embeds, _, _, missing_indexes = self.embed_image_query(query_image_feats, query_feature_map, bboxes)
        if query_embeds is None:
            return None, None
        query_embeds /= torch.linalg.norm(query_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        if cls is not None:
            return query_embeds, [item for index, item in enumerate(cls) if index not in missing_indexes]

        return query_embeds

    def forward_text(
            self,
            input_ids,
            attention_mask,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None, ):

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.model.config.return_dict

        text_embeds, text_outputs = self.model.owlvit.forward_text(input_ids=input_ids, attention_mask=attention_mask,
                                                                   output_attentions=output_attentions,
                                                                   output_hidden_states=output_hidden_states,
                                                                   return_dict=return_dict)

        text_embeds = text_embeds.unsqueeze(0)

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.unsqueeze(0)
        query_mask = input_ids[..., 0] > 0

        return text_embeds.to(self.device), query_mask.to(self.device)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> OwlViTObjectDetectionOutput:

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.return_dict

        # Embed images and text queries
        _, vision_outputs = self.model.owlvit.forward_vision(pixel_values=pixel_values,
                                                             output_attentions=output_attentions,
                                                             output_hidden_states=output_hidden_states,
                                                             return_dict=return_dict)

        # Get image embeddings
        last_hidden_state = vision_outputs[0]
        image_embeds = self.model.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )

        image_embeds = image_embeds.reshape(new_size)

        batch_size, num_patches, num_patches, hidden_dim = image_embeds.shape
        image_feats = torch.reshape(image_embeds, (batch_size, num_patches * num_patches, hidden_dim))

        # Predict object boxes
        pred_boxes = self.model.box_predictor(image_feats, image_embeds)

        (pred_logits, class_embeds) = self.model.class_predictor(image_feats, self.att_embeds.repeat(batch_size, 1, 1),
                                                                 self.att_query_mask)

        out = OwlViTObjectDetectionOutput(
            image_embeds=image_embeds,
            text_embeds=self.att_embeds,
            pred_boxes=pred_boxes,
            logits=pred_logits,
            class_embeds=class_embeds,
            vision_model_output=vision_outputs,
        )

        out.att_logits = out.logits  # TODO: remove later
        out.logits, out.obj = self.unk_head(out.logits)
        return out


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, model_name, pred_per_im=100, image_resize=768, device='cpu', method='regular'):
        super().__init__()
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.pred_per_im = pred_per_im
        self.method = method
        self.image_resize = image_resize
        self.device = device
        self.clip_boxes = lambda x, y: torch.cat(
            [x[:, 0].clamp_(min=0, max=y[1]).unsqueeze(1),
             x[:, 1].clamp_(min=0, max=y[0]).unsqueeze(1),
             x[:, 2].clamp_(min=0, max=y[1]).unsqueeze(1),
             x[:, 3].clamp_(min=0, max=y[0]).unsqueeze(1)], dim=1)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, viz=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        if viz:
            reshape_sizes = torch.Tensor([[self.image_resize, self.image_resize]]).repeat(len(target_sizes), 1)
            target_sizes = (target_sizes * self.image_resize / target_sizes.max(1, keepdim=True).values).long()
        else:
            max_values, _ = torch.max(target_sizes, dim=1)
            reshape_sizes = max_values.unsqueeze(1).repeat(1, 2)

        if self.method == "regular":
            results = self.post_process_object_detection(outputs=outputs, target_sizes=reshape_sizes)
        elif self.method == "attributes":
            results = self.post_process_object_detection_att(outputs=outputs, target_sizes=reshape_sizes)
        elif self.method == "seperated":
            results = self.post_process_object_detection_seperated(outputs=outputs, target_sizes=reshape_sizes)

        for i in range(len(results)):
            results[i]['boxes'] = self.clip_boxes(results[i]['boxes'], target_sizes[i])
        return results

    def post_process_object_detection(self, outputs, target_sizes=None):
        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, logits, boxes):
            scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
            topk_boxes = topk_indexes // logits.shape[2]
            labels = topk_indexes % logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob, logits, boxes)
        return results

    def post_process_object_detection_att(self, outputs, target_sizes=None):
        ## this post processing should produce the same predictions as `post_process_object_detection`
        ## but also report what are the most dominant attribute per class (used to produce some of the
        ## figures in the MS
        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob_att = torch.sigmoid(outputs.att_logits)

        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, logits, prob_att, boxes):
            scores, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), self.pred_per_im, dim=1)
            topk_boxes = topk_indexes // logits.shape[2]
            labels = topk_indexes % logits.shape[2]

            # Get the batch indices and prediction indices to index into prob_att
            batch_indices = torch.arange(logits.shape[0]).view(-1, 1).expand_as(topk_indexes)
            pred_indices = topk_boxes

            # Gather the attributes corresponding to the top-k labels
            # You will gather along the prediction dimension (dim=1)
            gathered_attributes = prob_att[batch_indices, pred_indices, :]

            # Now gather the boxes in a similar way as before
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            # Combine the results into a list of dictionaries
            dom_attr_idx = gathered_attributes.argmax(dim=-1)
            import ipdb;ipdb.set_trace()
            return [{'scores': s, 'labels': l, 'boxes': b, 'attributes': a} for s, l, b, a in
                    zip(scores, labels, boxes, gathered_attributes)]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob, logits, prob_att, boxes)
        return results

    def post_process_object_detection_seperated(self, outputs, target_sizes=None):
        ## predicts the known and unknown objects seperately. Used when the known and unknown classes are
        ## derived one from text and the other from images.

        logits, obj, boxes = outputs.logits, outputs.obj, outputs.pred_boxes
        prob = torch.sigmoid(logits)
        prob[..., -1] *= obj.squeeze(-1)

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        def get_known_objs(prob, out_logits, boxes):
            # import ipdb; ipdb.set_trace()
            scores, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.pred_per_im // 2, dim=1)
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        def get_unknown_objs(obj, out_logits, boxes):

            scores, topk_indexes = torch.topk(obj.unsqueeze(-1), self.pred_per_im // 2, dim=1)
            scores = scores.squeeze(-1)
            labels = torch.ones(scores.shape, device=scores.device) * out_logits.shape[-1]
            # import ipdb; ipdb.set_trace()
            boxes = torch.gather(boxes, 1, topk_indexes.repeat(1, 1, 4))
            return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = scale_fct.to(boxes.device)

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        boxes = boxes * scale_fct[:, None, :]

        results = get_known_objs(prob[..., :-1].clone(), logits[..., :-1].clone(), boxes)
        unknown_results = get_unknown_objs(prob[..., -1].clone(), logits[..., :-1].clone(), boxes)

        out = []
        for k, u in zip(results, unknown_results):
            out.append({
                "scores": torch.cat([k["scores"], u["scores"]]),
                "labels": torch.cat([k["labels"], u["labels"]]),
                "boxes": torch.cat([k["boxes"], u["boxes"]])
            })
        return out


def build2(args):
    device = torch.device(args.device)

    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.classnames_file}', 'r') as file:
        ALL_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.prev_classnames_file}', 'r') as file:
        PREV_KNOWN_CLASS_NAMES = sorted(file.read().splitlines())

    CUR_KNOWN_ClASSNAMES = [cls for cls in ALL_KNOWN_CLASS_NAMES if cls not in PREV_KNOWN_CLASS_NAMES]

    known_class_names = PREV_KNOWN_CLASS_NAMES + CUR_KNOWN_ClASSNAMES
    if args.unk_proposal and args.unknown_classnames_file != "None":
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.unknown_classnames_file}', 'r') as file:
            unknown_class_names = sorted(file.read().splitlines())
        unknown_class_names = [k for k in unknown_class_names if k not in known_class_names]
        unknown_class_names = [c.replace('_', ' ') for c in unknown_class_names]

    else:
        unknown_class_names = ["object"]

    if args.templates_file:
        with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.templates_file}', 'r') as file:
            templates = file.read().splitlines()

    else:
        templates = ["a photo of a {c}"]

    with open(f'data/{args.data_task}/ImageSets/{args.dataset}/{args.build_known_unknown_file}', 'r') as file:
        all_class_names = sorted(file.read().splitlines())
        new_class_names = [k for k in all_class_names if k not in known_class_names]

    model = FOMO(args, args.model_name, known_class_names, unknown_class_names,
                 templates, args.image_conditioned, device)

    postprocessors = PostProcess(args.model_name, args.pred_per_im, args.image_resize, device,
                                 method=args.post_process_method)
    return model, postprocessors
