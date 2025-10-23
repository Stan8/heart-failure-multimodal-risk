#!/usr/bin/env python
# coding: utf-8

# load BRAT data, a BERT tokenizer and model, and use it to encode entities
    
import sys
import os
import argparse
import logging
from collections import defaultdict
import json
# from functools import partial
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
from transformers import CamembertForMaskedLM
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch
# import evaluate
import transformers
# from transformers.tokenization_xlm_roberta import SPIECE_UNDERLINE
SPIECE_UNDERLINE = "▁"

# brat annotations
import load_brat_data_no_spacy

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm

# Program version
version = '0.1'

logging.basicConfig(
    # level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# https://stackoverflow.com/questions/68277801/extracting-meaningful-error-message-from-runtimeerror-cuda-error-device-side
CUDA_LAUNCH_BLOCKING = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = CUDA_LAUNCH_BLOCKING

# https://stackoverflow.com/questions/66984523/how-to-change-from-conll-format-into-a-sentences-list
# https://huggingface.co/transformers/v3.2.0/custom_datasets.html
from pathlib import Path
import re

def parse_execute_command_line():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)

    groupIO = parser.add_argument_group('Inputs and outputs')
    groupIO.add_argument(
        "-i", "--input-directory",
        required=True,
        help="input corpus directory: all BRAT files in that directory",
    )
    groupIO.add_argument(
        "-m", "--model-checkpoint",
        required=True,
        default="camembert-base",
        help="pretrained Hugging Face library model checkpoint to fine-tune"
        ". Default=%(default)s"
    )
    groupIO.add_argument(
        "-o", "--output-file",
        default="results",
        help="TSV file in which encodings are recorded"
        ". Default=%(default)s",
    )
    groupIO.add_argument(
        "-l", "--logging-directory",
        default="logs",
        help="directory in which logs are recorded"
        ". Default=%(default)s",
    )
    groupOpt = parser.add_argument_group('Options')
    groupOpt.add_argument(
        "--entity-types",
        default=['AGE', 'PATHOLOGIE', 'SIGNE_SYMPTOME', 'TRAITEMENT', 'ANATOMIE', 'EXAMEN', 'ENTOURAGE', 'AUTONOMIE', 'CONCENTRATION', 'MODE', 'DOSE', 'FREQUENCE', 'PARAMETRE_MESURABLE', 'VALEUR', 'NEGATION', 'HYPOTHETIQUE', 'EVOLUTION_TRAITEMENT_PARAMETRE', 'COMPORTEMENT', 'DUREE', 'EVOLUTION', 'CHANGEMENT_LIEU', 'LIEU', 'DATE', 'HEURE'],
        nargs='+',
        type=str,
        help="list of entity types to use for encoding"
        ". Default=%(default)s",
    )
    groupOpt.add_argument(
        "--mention-aggregation",
        default='average',
        type=str,
        help="mode of aggregation of mentions: average, max (nyi)"
        ". Default=%(default)s",
    )
    groupOpt.add_argument(
        "--wordpiece-aggregation",
        default='average',
        type=str,
        help="mode of aggregation of word pieces: average, max (nyi)"
        ". Default=%(default)s",
    )
    groupOpt.add_argument(
        "-s", "--stride",
        default=64,
        type=int,
        help="number of tokens in overlap of two chunks of text"
        ". Default=%(default)s",
    )
    groupOpt.add_argument(
        "-S", "--seed",
        default=42,
        type=int,
        help="random seed for training"
        ". Default=%(default)s",
    )

    groupS = parser.add_argument_group('Special')
    groupS.add_argument("-q", "--quiet", action="store_true", help="suppress reporting progress info.")
    groupS.add_argument("--debug", action="store_true", help="print debug info.")
    groupS.add_argument("-v", "--version", action="version", version='%(prog)s ' + version, help="print program version.")

    args = parser.parse_args()
    return args

def main(args):

    logger = logging.getLogger()
    if not args.quiet:
        logger.setLevel(logging.INFO)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # output_dir = args.output_directory
    # create_dir(output_dir)
    logging_dir = args.logging_directory
    create_dir(logging_dir)

    # Load BRAT-annotated corpus
    logging.info(f"Loading BRAT corpus from '{args.input_directory}'...")
    docs = load_brat_annotations(args.input_directory)
    logging.info(f"Read {len(docs.documents)} documents")

    stride = args.stride
    entity_types = args.entity_types
    model_checkpoint = args.model_checkpoint
    logging.info(f"entity_types: {entity_types}")

    logging.info(f"Transformers version: {transformers.__version__}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"Running model on device '{device}'")

    # Load model's tokenizer and tokenize into encoded subwords -> corpus->{train, val}->{tokenized_text}
    logging.info(f"Loading tokenizer for model '{model_checkpoint}'")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    logging.info(f"Loading CamembertForMaskedLM for model '{model_checkpoint}'")
    model = CamembertForMaskedLM.from_pretrained(model_checkpoint)
    logging.info(f"Loading model to device {device}")
    model = model.to(device)    # is it ok to load model to GPU once and for all?
    logging.info(f" -> done")

    emb_dim = model.roberta.encoder.layer[-1].output.dense.out_features
    # null_embedding = torch.full((emb_dim,), -10)
    null_embedding = torch.full((emb_dim,), 0.0).to(device)
    docs_strings = []
    docs_encodings = []
    docs_cls_encodings = []
    doc_files = [doc.file_name for doc in docs.documents]
    # with open(args.output_file, "w", encoding="utf-8") as fs:
    # print("[", file=fs)
    for doc in docs.documents:
        logging.info(f"Splitting text with tokenizer model of '{model_checkpoint}'")
        tokenizer_output = prepare_tokens(
            tokenizer,
            [doc.text],
            stride=stride,
            padding='max_length',
            add_special_tokens=True,
            device=device
        )

        logging.info(f"Encoding text with model of '{model_checkpoint}'")
        model_output = bert_compute_model_output(
            tokenizer_output,
            model,
            device=device
        )
        output_embeddings = model_output.hidden_states[-1]

        logging.info(f"Encoding [CLS] with model of '{model_checkpoint}'")
        cls_encoding = average_cls_embeddings(output_embeddings, device=device)
        logging.info(f"  -> {cls_encoding.shape}")
        docs_cls_encodings.append(cls_encoding)

        logging.info(f"Getting entity mention spans then strings for '{entity_types}'")
        entity_type_spans = brat_doc_get_spans_by_type(doc, entity_types)
        # for information
        entity_type_strings = brat_doc_get_mentions_by_type(doc, entity_types)
        logging.info(f"  entity mention strings: '{entity_type_strings}'")
        logging.info(f"Getting entity mention chunk-token spans for '{entity_types}'")
        entity_type_ctspans = [
            [
                chunk_token_span_for_character_span(
                    tokenizer_output['offsets'],
                    tokenizer_output['offset_mapping'],
                    span,
                    add_special_tokens=True
                )
                for span in spans
            ]
            for entity_type, spans in zip(entity_types, entity_type_spans)
        ]

        logging.info(f"Encoding entity mention token spans for {entity_types} with tokenizer and model of '{model_checkpoint}'")
        # torch.tensor([n_entity_types, embedding_size])
        # do not stack now, first perform imputation of null values
        # n_entity_types × torch.tensor([embedding_size])
        '''
        entity_type_encodings = torch.stack( # should it be row_stack(transpose) instead?
            [
                (aggregated_ct_span_embeddings(
                    output_embeddings,
                    ct_spans,
                    wordpiece_aggregation=args.wordpiece_aggregation,
                    mention_aggregation=args.mention_aggregation,
                    device=device,
                ) if len(ct_spans) > 0 else null_embedding)
            for entity_type, ct_spans in zip(entity_types, entity_type_ctspans)
            ]
        )
'''
        entity_type_encodings = torch.stack(  #fix for None

            [
                (
                    aggregated_ct_span_embeddings(
                        output_embeddings,
                        [span for span in ct_spans if span is not None],  # filter out None
                        wordpiece_aggregation=args.wordpiece_aggregation,
                        mention_aggregation=args.mention_aggregation,
                        device=device,
                    ) if any(span is not None for span in ct_spans) else null_embedding
                )
                for entity_type, ct_spans in zip(entity_types, entity_type_ctspans)
            ]
        )

        # logging.info(f"Mean imputation of missing values in entity encodings for {n_entity_types} entity types")
        # entity_type_encodings = simple_imputation(entity_type_encodings, null_embedding)
        docs_strings.append(entity_type_strings)
        # n_docs × n_entity_types × torch.tensor([embedding_size])
        docs_encodings.append(entity_type_encodings)
        # logging.info(f"Saving encodings for document {doc.file_name}")
        # write_doc_encodings(
        #     doc.file_name,
        #     entity_type_strings,
        #     entity_type_encodings,
        #     entity_types,
        #     cls_encoding,
        #     fs
        # )
    # print("]", file=fs)

    logging.info(f"Dimensions of one doc_encoding: {docs_encodings[0].shape}")
    docs_encodings = torch.stack(docs_encodings)
    logging.info(f"Dimensions of doc_encodings: {docs_encodings.shape}")

    logging.info(f"Writing encodings of shape {docs_encodings[0].shape} for {len(entity_types)} entity types for {len(doc_files)} documents to JSON file {args.output_file}")
    save_encodings_pt(doc_files, docs_strings, docs_encodings, entity_types, docs_cls_encodings, args.output_file)

    return

#================
# Simple imputation of missing values, with tensors
#================

def simple_imputation(X, null_embedding):
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # return imp.fit_transform(X)
    # X: n_docs × n_entity_types × (torch.tensor([embedding_size]) or np.nan)
    return average_imputation_n_docs(X, null_embedding)

def average_imputation_n_docs(docs_encodings, null_emb):
    """Given a tensor of n_docs * m_entity_types * emb_dim
    and the null vector null_emb (emb_dim),

    perform average imputation of null values for each entity_type individually.
    Returns the list of entity_type indices (in 0..m_entity_types-1)
    for which at least one document has a non-null value."""
    n_docs, m_entity_types, emb_dim = docs_encodings.shape
    non_null = [] # torch.full([m_entity_types], True)
    for e in range(m_entity_types):
        logging.info(f"Average imputation for entity_type {e}/{m_entity_types}")
        avg = average_imputation_of_missing_values(docs_encodings[:, e, :], null_emb)
        if avg is not None:
            # non_null[e] = False
            # non_null.append(e)
        # else:
            docs_encodings[:, e, :] = avg
            non_null.append(e)
    return non_null

def average_imputation_of_missing_values(n_emb, null_emb):
    """Given a tensor of n_doc vectors of dim(emb_dim)
    and the null vector null_emb of dim(emb_dim),

    compute the average of the non-null vectors
    and insert it in place of each null vector"""
    mask = (n_emb != null_emb)[:, 0]
    if mask.sum().item() == 0:
        return None
    avg = n_emb[mask].mean(0)
    n_emb[(mask == False)] = avg
    return n_emb


# tokenize text and compute span indexes
def prepare_tokens(tokenizer_fn, texts, max_length=512, stride=0, padding=False, add_special_tokens=True, device="cpu"):
    tokenizer_output = tokenizer_fn(
        texts,
        is_split_into_words=False,
        # return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
        truncation=True,  # important to split text into max_length chunks
        return_overflowing_tokens=True,  # important to split text into max_length chunks
        max_length=max_length,
        stride=stride,
        padding=padding,
        return_offsets_mapping=True,
        return_tensors="pt",
    ).to(device)
    tokenizer_plus = {}
    tokenizer_output['tokens'] = [
        tokenizer_fn.convert_ids_to_tokens(input_ids)
        for input_ids in tokenizer_output['input_ids']
    ]
    tokenizer_output['offsets'] = chunk_offsets(
        tokenizer_output['offset_mapping'],
        tokenizer_output['attention_mask'],
        add_special_tokens
    )
    
    return tokenizer_output

def chunk_offsets(offset_mapping, attention_mask, add_special_tokens):
    """Computes the offsets of each chunk in offset_mapping

    offset_mapping : list of N chunks that make up a text,
    split into small enough chunks,
    where a chunk is a list of (start, end) character offsets of tokens
    where end is the offset of the last character of a token (+1)
    
    attention_mask: tensor of shape(N),
    contains zeros in padding positions, else ones

    add_special_tokens : if True, each chunk starts and ends with a special token

    Returns a list of (start_offset, end_offset) tuples containing
    the (zero-based) start and end character offsets of each chunk
    """
    offset_l = []
    for i, offsets in enumerate(offset_mapping):
        mask = attention_mask[i]
        if add_special_tokens:
            offsets = offsets[1:-1]
            mask = mask[1:-1]
        last_non_masked = mask.sum()-1 # subtracting 1 seems necessary
        offsets = offsets[:last_non_masked]
        # print(f"i={i}, add_special_tokens={add_special_tokens}, last_non_masked={last_non_masked}, offsets.shape={offsets.shape}, offsets[0][0]={offsets[0][0]}, offsets[-1][1]={offsets[-1][1]}")
        offset_l.append([offsets[0][0], offsets[-1][1]])
    return offset_l

# run model inference to obtain output encodings
def bert_compute_model_output(tokenizer_output, model, device="cpu"):
    keywords = ['input_ids',
                'attention_mask',
                 # 'offset_mapping',
                 # 'overflow_to_sample_mapping',
                 # 'tokens',
                 # 'offsets',
               ]
    tok_out = {k: tokenizer_output[k] for k in keywords}
    with torch.no_grad():
        model_output = model(**tok_out, output_hidden_states=True)
    return model_output


#================
# use tensor products to compute summary embedding for a set of entity mentions
#================

def aggregated_ct_span_embeddings(
        embeddings,
        ct_spans,
        wordpiece_aggregation='average', # TODO: max?
        mention_aggregation='average',   # TODO: max?
        device='cpu',
):
    """
    How to compute one embedding for a collection of entity mention spans.

    Compute mention embeddings
    - as the mean of token embeddings
    - for all mentions in parallel
    - in all text chunks in parallel,
    - then compute the mean of all these mention embeddings,
    - all of this with tensor operations.

    Input:
    embeddings : torch.tensor([n_chunks, m_tokens, embedding_dim])
        a text split into chunks

    ct_spans : ((chunk_id, start_token, end_token), ...)
        p_spans mention indices into chunks and tokens

    wordpiece_aggregation : average
        how token embeddings are combined to compute a mention embedding

    mention_aggregation : average
        how mention embeddings are combined to compute an aggregate embedding
        for all the input mentions

    Output:
    embedding : torch.tensor([embedding_dim])
    """
    
    n_chunks, m_toks, emb_dim = embeddings.shape
    # print(f"aggregated_ct_span_embeddings: {n_chunks} chunks of {m_toks} tokens of embedding dimension {emb_dim}")
    # print("\nEmbeddings\n", embeddings.shape, "\n", embeddings)

    p_spans = len(ct_spans)
    # print(f"\n{p_spans} spans: {ct_spans}")
    logging.info(f"aggregated_ct_span_embeddings: {p_spans} spans in {n_chunks} chunks of {m_toks} tokens of embedding dimension {emb_dim}")

    av_tok_emb_op = torch.zeros((n_chunks, p_spans, m_toks)).to(device)
    for i, (c, b, e) in enumerate(ct_spans):
        av_tok_emb_op[c, i, b:e] = 1/(e-b)
    # print("\nAverage token embedding operator\n", av_tok_emb_op.shape, "\n", av_tok_emb_op)

    # print(f"\nMatrix multiplication of token embedding operator {av_tok_emb_op.shape} by embeddings {embeddings.shape}:")
    av_tok_emb_per_chunk = torch.matmul(av_tok_emb_op, embeddings)
    # print("\nAverage token embeddings\n", av_tok_emb_per_chunk.shape, "\n", av_tok_emb_per_chunk)

    av_tok_emb = av_tok_emb_per_chunk.sum(dim=0)
    # print("\nAverage token embeddings\n", av_tok_emb.shape, "\n", av_tok_emb)

    av_emb_op = torch.full((p_spans,), 1/p_spans).to(device)
    # print("\nAverage embedding operator\n", av_emb_op.shape, "\n", av_emb_op)
    av_mention_emb = torch.matmul(av_emb_op, av_tok_emb)
    # print("\nTensor multiply\n", av_mention_emb.shape, "\n", av_mention_emb)

    return av_mention_emb


#================
# BERT encoder for [CLS]
#================

# use tensor multiplication to compute mean CLS embedding across chunks
def average_cls_embeddings(embeddings, device='cpu'): # torch.tensor([n_chunks, n_token, emb_dim])
    logging.info(f"embeddings {embeddings.shape}")
    cls_e = embeddings[:, 0]    # torch.tensor([n_chunks, emb_dim])
    logging.info(f"cls_e {cls_e.shape}")
    n_chunks, emb_dim = cls_e.shape
    av_chunk_op = torch.full((n_chunks,), 1/n_chunks).to(device) # torch.tensor([n_chunks])
    logging.info(f"av_chunk_op {av_chunk_op.shape}")
    av_cls_e = torch.matmul(av_chunk_op, cls_e)
    logging.info(f"av_cls_e {av_cls_e.shape}")
    return av_cls_e             # torch.tensor([emb_dim])

#================
# Convert character offsets to token offsets
#================

def chunk_token_span_for_character_span(offsets, offset_mapping, span, add_special_tokens):
    """Given chunk offsets list and token offset list,
    and a character span for an entity mention,
    returns the first found chunk index and token index span for that character span,
    or None if none found."""
    ct_spans = span2chunkindex(
        offsets,
        offset_mapping,
        span,
        add_special_tokens
    )
    # assumes at least one chunk contains the complete span
    complete_spans = [cbe for cbe in ct_spans if all(i is not None for i in cbe)]
    if len(complete_spans) > 0:
        return complete_spans[0] # use only first complete span
    else:
        return None

# obtain chunk+token index for span (b, e)
def span2chunkindex(chunk_offsets_l, token_offsets_l, span, add_special_tokens):
    """Given chunk offsets list and token offset list,
    and a character span for an entity mention,
    returns the (possibly multiple) chunk index and token index span
    for that character span.
    
    chunk_offsets_l : list of length N_chunks found in tokenizer_output 'offsets'.
      Each element i has the form
      (chunk_i_begin, chunk_i_end) as character offsets
    
    token_offsets_l : list of length N_chunks found in tokenizer_output 'offset_mapping'.
      Each element i has the form
      ((token_begin, token_end), ...) as character offsets
      of length N_tokens_i
    
    span = (b, e): character offsets of given span
    
    add_special_tokens : if True, ignores the start and end special tokens of the chunk
    
    Output:
    
    A list of chunk-token spans. Each element has the form
    ((chunk_i, token_it), (chunk_j, token_ju))
    where chunk_i is a chunk index (starting with 0) (max i = N_chunks-1)
    and token_it is a token index (starting with 0) in chunk_i (max it = len(chunk_i)-1)
    """

    # print(f"span2chunkindex: {span}")
    b, e = span
    ct_spans = []
    for i, (c, token_offsets) in enumerate(zip(chunk_offsets_l, token_offsets_l)):
        base_index = 0
        if add_special_tokens:
            token_offsets = token_offsets[1:-1]
            base_index += 1

        token_b_i, token_e_i = None, None
        if b >= c[0] and b < c[1]: # chunk contains span begin
            for j, token in enumerate(token_offsets):
                if b >= token[0] and b < token[1]:
                    token_b_i = j
                    # start from that begin token to look for end below
                    token_offsets = token_offsets[token_b_i:]
                    token_b_i += base_index
                    break
        if e > c[0] and e <= c[1]: # chunk contains span end
            for j, token in enumerate(token_offsets):
                if e < token[0]: # been beyond e: stop
                    break
                if e >= token[0] and e <= token[1]:
                    token_e_i = (
                        token_b_i if token_b_i is not None
                        else 0
                    ) + j + 1 # token_b_i integrates base_index; add 1 for next position
                    # break # not yet: because of zero-width span tokens,
                    # there might be multiple tokens at end
        if any(t is not None for t in (token_b_i, token_e_i)):
            ct_spans.append([i, token_b_i, token_e_i])
    return ct_spans

#================
# Load and obtain BRAT entity annotations
#================

def load_brat_annotations(data_path, language='fr'):
    # data_path = "/home/pz/SHARE/RESSOURCES/LOGICIELS/TAL/Annotation/BRAT/brat-v1.3_Crunchy_Frog/data/predhic/guide-annotation-v4.0-20230705/"
    # language: for determining the tokenizer and distinguishing plot names
    return load_brat_data_no_spacy.main(
        data_path,
        language=language,
        verbose=False
    )

def brat_doc_get_spans_by_type(doc, entity_types):
    spans_l = []
    mentions = get_mentions_by_type(doc)
    return [
        ([(e.start, e.end) for e in mentions[t]] if t in mentions
         else [])
        for t in entity_types
    ]

def brat_doc_get_mentions_by_type(doc, entity_types):
    text_l = []
    mentions = get_mentions_by_type(doc)
    return [
        ([e.entity_string for e in mentions[t]] if t in mentions
         else [])
        for t in entity_types
    ]


def get_mentions_by_type(doc):
    mentions = defaultdict(list)
    for e in doc.entities:
        mentions[e.entity_type].append(e)
    return mentions

#================
# Write results
#================

# TODO
def write_doc_encodings(
        doc_file,              # doc
        mentions,               # n_entity_types x n_mentions
        encodings,              # torch.tensor([n_entity_types, embedding_dim])
        entity_types,           # n_entity_types
        cls_encoding,          # torch.tensor([embedding_dim])
        output_fh               # file handle to write to
):
    """Create table with filename
    and for each entity_type, list of mentions and list of encodings
    then save it to json file"""

    logging.info(f"Writing [CLS] encoding ({cls_encoding.shape}) and {len(entity_types)} entity_type encodings ({encodings.shape}) for document '{doc_file}' to JSON file")

    logging.info(f"[CLS]: " +
                 f"{len(cls_encoding)} cls_encoding " +
                 f"of shape {cls_encoding.shape}")

    data = {                    # initialize dataset
        'file': doc_file,
        'CLS': cls_encoding.tolist(), # convert tensor to list
    }
    for i, t in enumerate(entity_types): # add entity encodings and names
        data[t+"_"] = mentions[i] # mention list for doc for one entity_type
        data[t] = encodings[i].tolist() # encoding for doc for one entity_type, convert tensor to list
        logging.info(f"entity_type {i} ({t}): " +
                     f"{len(data[t+'_'])} mentions " +
                     f"of encoding size {len(data[t])}")
    logging.info(f"Writing data to JSON file")
    json.dump(data, output_fh, indent=0) # insert \n instead of spaces
    print(",", file=output_fh, flush=True)
    return

def save_encodings(
        doc_files,              # n_docs
        mentions,               # n_docs x n_entity_types x n_mentions
        encodings,              # torch.tensor([n_docs, n_entity_types, embedding_dim])
        entity_types,           # n_entity_types
        cls_encodings,          # n_docs x torch.tensor([embedding_dim])
        output_file
):
    """Create table with filename
    and for each entity_type, list of mentions and list of encodings
    then save it to json file"""

    assert len(doc_files) == len(mentions), f"{len(doc_files)} documents but {len(mentions)} mention lists"
    assert len(doc_files) == len(encodings), f"{len(doc_files)} documents but {len(encodings)} encoding lists"
    assert len(doc_files) == len(cls_encodings), f"{len(doc_files)} documents but {len(cls_encodings)} cls_encoding lists"

    logging.info(f"Writing encodings of shape {encodings[0].shape} for {len(entity_types)} entity types for {len(doc_files)} documents to JSON file {output_file}")

    logging.info(f"[CLS]: " +
                 f"{len(cls_encodings)} cls_encoding " +
                 f"of shape {cls_encodings[0].shape}")

    data = {                    # initialize dataset
        'file': doc_files,
        'CLS': [e.tolist() for e in cls_encodings], # convert tensor to list
    }
    for i, t in enumerate(entity_types): # add entity encodings and names
        data[t+"_"] = []
        data[t] = []
        for m, e in zip(mentions, encodings): # for each document
            data[t+"_"].append(m[i]) # mention list for doc for one entity_type
            data[t].append(e[i].tolist()) # encoding for doc for one entity_type, convert tensor to list
        logging.info(f"entity_type {i} ({t}): " +
                     f"{len(data[t])} mention lists and encodings " +
                     f"of length {len(data[t][0])}")
    df = pd.DataFrame(
        data=data,
        # dtype={
        #     'file': str,
        #     'TRAITEMENT_s': list,  # mention strings
        #     'PATHOLOGIE_s': list,
        #     'TRAITEMENT': list,    # encodings
        #     'PATHOLOGIE': list,
        #     ...
        # }
    )
    logging.info(f"Writing data to JSON file {output_file}")
    df.to_json(output_file, index=False)
    return

def save_encodings_pt(
        doc_files,              # n_docs
        mentions,               # n_docs x n_entity_types x n_mentions
        encodings,              # torch.tensor([n_docs, n_entity_types, embedding_dim])
        entity_types,           # n_entity_types
        cls_encodings,          # n_docs x torch.tensor([embedding_dim])
        output_file
):
    """Create dict with filename
    and for each entity_type, list of mentions and list of encodings
    then save it to pickle file"""

    assert len(doc_files) == len(mentions), f"{len(doc_files)} documents but {len(mentions)} mention lists"
    assert len(doc_files) == len(encodings), f"{len(doc_files)} documents but {len(encodings)} encoding lists"
    assert len(doc_files) == len(cls_encodings), f"{len(doc_files)} documents but {len(cls_encodings)} cls_encoding lists"

    logging.info(f"Writing encodings of shape {encodings[0].shape} for {len(entity_types)} entity types for {len(doc_files)} documents to JSON file {output_file}")

    logging.info(f"[CLS]: " +
                 f"{len(cls_encodings)} cls_encoding " +
                 f"of shape {cls_encodings[0].shape}")

    data = {                    # initialize dataset
        'file': doc_files,
        'CLS': [e.tolist() for e in cls_encodings], # convert tensor to list
    }
    for i, t in enumerate(entity_types): # add entity encodings and names
        data[t+"_"] = []
        data[t] = []
        for m, e in zip(mentions, encodings): # for each document
            data[t+"_"].append(m[i]) # mention list for doc for one entity_type
            data[t].append(e[i].tolist()) # encoding for doc for one entity_type, convert tensor to list
        logging.info(f"entity_type {i} ({t}): " +
                     f"{len(data[t])} mention lists and encodings " +
                     f"of length {len(data[t][0])}")
    logging.info(f"Writing data to pt file {output_file}")
    torch.save(data, output_file)
    return

#================
# Utils
#================

def create_dir(d):
    if not os.path.isdir(d):
        dd = os.path.dirname(d) # recurse as needed
        if dd != '':
            create_dir(dd)
        os.mkdir(d)
    return

if __name__ == '__main__':
    main(parse_execute_command_line())
