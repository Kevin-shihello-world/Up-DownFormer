
from transformers import BertConfig as config
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
import math

from torch.nn.parameter import Parameter

###x_online = torch.ones(128, 128)#
class BertSelfAttention_new_not_1(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)##Now use to do on offline informs to get informs from x_online(for in Bert config.hidden_size 
        #= self.all_head_size so we can use it like this)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)##Now use to do on online informs to deal with its informs and to give informs to the offline informs
        self.value = nn.Linear(config.hidden_size, self.all_head_size)##Now use to do on offline informs to deal with its informs
        self.weight_1 = nn.Linear(24, 24)
        self.edgefeats_updater = nn.Linear(self.num_attention_heads, self.num_attention_heads)
        
        self.nodefeats_reshaper = nn.AdaptiveAvgPool1d(128)
        self.edgefeats_reshaper_0 = nn.AdaptiveAvgPool1d(128*6)
        self.edgefeats_reshaper = nn.AdaptiveAvgPool1d(128*11)
        self.edgefeats_user = nn.Linear(128*2, 128)
        self.reshaper = nn.AdaptiveAvgPool2d((128, 128*12))
        self.edgefeats_user.weight = Parameter(self.reshaper(self.key.weight.t().unsqueeze(0) ).squeeze(0) )
        #self.nodesfeats_updater = nn.Linear(128*12, 128*12)
        #self.nodesfeats_updater.weight = Parameter(self.reshaper(self.query.weight.t().unsqueeze(0) ).squeeze(0) )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        #print("edgefeats_user:", self.edgefeats_user.weight.shape)
        hidden_states_ = hidden_states + self.query(hidden_states)
        global edge_online_c
        global edge_online_f
        global x_online
        edge_online_f_ = edge_online_f
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        ####
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length = self.attention_head_size
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", x_online, positional_embedding)
                edge_online_f_ = edge_online_f_ + relative_position_scores
        ####
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
            
            
            
        #print("edge_online_f_:", edge_online_f_.shape)
        #print("edge_online_c:", edge_online_c.shape)
        edge_online_on___ = edge_online_f_@edge_online_c
        edge_online_f_ = edge_online_f_.permute(0, 2, 3, 1)##[batch, sentence_length, token_informs_length, edge_informs]
        shape_0 = edge_online_f_.shape[0]
        shape_1 = edge_online_f_.shape[1]
        shape_2 = edge_online_f_.shape[2]
        edgefeats_reshape_back_0 = nn.AdaptiveAvgPool1d(12*shape_1)
        edge_online_f_update = edgefeats_reshape_back_0(self.value(self.edgefeats_reshaper_0(edge_online_f_.reshape(shape_0, shape_1, 12*shape_1) ) ) )
        edge_online_f = edge_online_f + edge_online_f_update.reshape(shape_0, shape_1, shape_1, 12).permute(0, 3, 1, 2)

        edge_online_on__ = torch.cat((edge_online_f, self.nodefeats_reshaper(hidden_states_)), dim = 2)
        edge_online_on_ = self.edgefeats_user(edge_online_on__)
       #print("edge_online_on_:", edge_online_on_.shape)    
        edgefeats_reshape_back_1 = nn.AdaptiveAvgPool1d(shape_1)
        edge_online_on = edgefeats_reshape_back_1(edge_online_on_)#.reshape(shape_0, shape_1, shape_1)
       #print("edge_online_f:", edge_online_f.shape)
       #print("edge_online_on:", edge_online_on.shape)
        #edge_online_on = edge_online_on_.narrow(2, 0, shape_2)
        #print(edge_online_on.shape)
        ##x_online = x_online + torch.matmul(edge_online_on, x_online)
        x_online_updater = self.key(x_online)
        x_online = x_online_updater@x + (1 - x_online_updater)@x_online
        x_online_1 = x_online
        
        '''
        ####norm = torch.sigmoid(torch.norm(hidden_states_, p=2, dim=2) )#.unsqueeze(0)
        ##print("3:", x_online_1.shape)
        shape_0 = hidden_states.shape[0]
        shape_1 = hidden_states.shape[1]
        x_test = torch.reshape(hidden_states_, (shape_0, shape_1, 32, 24) ) 
        x_online_2 = torch.reshape(x_online_1, (shape_0, shape_1, 24, 32))
        p2pattention__ = torch.matmul(x_test, x_online_2)
        p2pattention_ = p2pattention__.narrow(3, 0, 24)
        #print("norm:", norm.shape)
        #print("x_online_1:", x_online_1.shape)
        p2pattention = torch.reshape(p2pattention_, (shape_0, shape_1, 768))
        norm_1 = torch.sigmoid(torch.norm(p2pattention, p=2, dim=2) )
        
        x_online = 1/2*(torch.matmul(torch.diag_embed(norm), hidden_states_) + x_online_1)
        outputs = 1/2*torch.squeeze(torch.matmul(torch.diag_embed(norm_1), x_online_1) + hidden_states_, 0)
        ####'''
        
        x_updater = self.query(x)
        x = x_updater@x_online + (1 - x_updater)@x
        outputs = (x, edge_online_c) if output_attentions else (outputs,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
class BertSelfAttention_new_lastlayer(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)##Now use to do on offline informs to get informs from x_online(for in Bert config.hidden_size 
        #= self.all_head_size so we can use it like this)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)##Now use to do on online informs to deal with its informs and to give informs to the offline informs
        self.value = nn.Linear(config.hidden_size, self.all_head_size)##Now use to do on offline informs to deal with its informs
        self.weight_1 = nn.Linear(24, 24)
        self.edgefeats_updater = nn.Linear(self.num_attention_heads, self.num_attention_heads)
        
        self.nodefeats_reshaper = nn.AdaptiveAvgPool1d(128)
        self.edgefeats_reshaper_0 = nn.AdaptiveAvgPool1d(128*6)
        self.edgefeats_reshaper = nn.AdaptiveAvgPool1d(128*11)
        self.edgefeats_user = nn.Linear(128*2, 128)
        self.reshaper = nn.AdaptiveAvgPool2d((128, 128*12))
        self.edgefeats_user.weight = Parameter(self.reshaper(self.key.weight.t().unsqueeze(0) ).squeeze(0) )
        #self.nodesfeats_updater = nn.Linear(128*12, 128*12)
        #self.nodesfeats_updater.weight = Parameter(self.reshaper(self.query.weight.t().unsqueeze(0) ).squeeze(0) )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        #print("edgefeats_user:", self.edgefeats_user.weight.shape)
        hidden_states_ = hidden_states + self.query(hidden_states)
        global edge_online_c
        global edge_online_f
        global x_online
        edge_online_f_ = edge_online_f
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        ####
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length = self.attention_head_size
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", x_online, positional_embedding)
                edge_online_f_ = edge_online_f_ + relative_position_scores
        ####
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
            
            
            
        #print("edge_online_f_:", edge_online_f_.shape)
        #print("edge_online_c:", edge_online_c.shape)
        edge_online_on___ = edge_online_f_
        edge_online_f_ = edge_online_f_.permute(0, 2, 3, 1)##[batch, sentence_length, token_informs_length, edge_informs]
        shape_0 = edge_online_f_.shape[0]
        shape_1 = edge_online_f_.shape[1]
        shape_2 = edge_online_f_.shape[2]
        edgefeats_reshape_back_0 = nn.AdaptiveAvgPool1d(12*shape_1)
        edge_online_f_update = edgefeats_reshape_back_0(self.value(self.edgefeats_reshaper_0(edge_online_f_.reshape(shape_0, shape_1, 12*shape_1) ) ) )
        edge_online_f = edge_online_f + edge_online_f_update.reshape(shape_0, shape_1, shape_1, 12).permute(0, 3, 1, 2)
        
        edge_online_on__ = torch.cat((edge_online_f[0], self.nodefeats_reshaper(hidden_states_)), dim = 2)
        edge_online_on_ = self.edgefeats_user(edge_online_on__)
       #print("edge_online_on_:", edge_online_on_.shape)                 
        edgefeats_reshape_back_1 = nn.AdaptiveAvgPool1d(shape_1)
        edge_online_on = nn.functional.softmax(edgefeats_reshape_back_1(edge_online_on_), dim=-1)##on
       #print("edge_online_f:", edge_online_f.shape)
       #print("edge_online_on:", edge_online_on.shape)
        #edge_online_on = edge_online_on_.narrow(2, 0, shape_2)
        #print(edge_online_on.shape)
        x_online = x_online + torch.matmul(edge_online_on, x_online)
        x_online = self.key(x_online) + x_online
        x_online_1 = x_online
        
        #hidden_states += self.query(hidden_states)
        ##print("1:",x_online.shape)
        
        ##print("2:", x_online_1.shape)
        '''norm = torch.sigmoid(torch.norm(hidden_states_, p=2, dim=2) )#.unsqueeze(0)
        ##print("3:", x_online_1.shape)
        shape_0 = hidden_states.shape[0]
        shape_1 = hidden_states.shape[1]
        x_test = torch.reshape(hidden_states_, (shape_0, shape_1, 32, 24) ) 
        x_online_2 = torch.reshape(x_online_1, (shape_0, shape_1, 24, 32))
        p2pattention__ = torch.matmul(x_test, x_online_2)
        p2pattention_ = p2pattention__.narrow(3, 0, 24)
        #print("norm:", norm.shape)
        #print("x_online_1:", x_online_1.shape)
        p2pattention = torch.reshape(p2pattention_, (shape_0, shape_1, 768))
        norm_1 = torch.sigmoid(torch.norm(p2pattention, p=2, dim=2) )
        
        x_online = 1/2*(torch.matmul(torch.diag_embed(norm), hidden_states_) + x_online_1)
        outputs = 1/2*torch.squeeze(torch.matmul(torch.diag_embed(norm_1), x_online_1) + hidden_states_, 0)'''
        
        

        ####
        x_updater = self.query(x)
        x = x_updater@x_online + (1 - x_updater)@x
        outputs = (x, edge_online_c) if output_attentions else (outputs,)
        ##outputs = (outputs, edge_online_c) if output_attentions else (outputs,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    
class BertSelfAttention_new_1(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        #self.edgefeats_adder = nn.Linear(self.num_attention_heads, self.attention_head_size)##add edge feats to Up-Down GNN
        #self.edgefeats_updater = nn.Linear(self.num_attention_heads, self.num_attention_heads)
        #self.edgefeats_user = nn.Linear(self.num_attention_heads, self.num_attention_heads)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        global x_online
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        global edge_online_c
        global edge_online_f
        edge_online_c = attention_probs
        edge_online_f = 10*torch.sigmoid(1/10*attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        global x_online
        x_online = context_layer
        #print("context_layer:", context_layer.shape)
        #print("hidden_states:", hidden_states.shape)
        context_layer = context_layer.view(new_context_layer_shape)
        #print("context_layer:", context_layer.shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs





