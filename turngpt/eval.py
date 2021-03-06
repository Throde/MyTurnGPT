from argparse import ArgumentParser
from logging import error
import math
import torch
from torch.serialization import save
from tqdm import tqdm
from os.path import split, join
from os import makedirs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl

from ttd.basebuilder import add_builder_specific_args
from ttd.utils import write_txt, fscores
from ttd.tokenizer_helpers import convert_ids_to_tokens

from turngpt.turngpt_dm import TurnGPTDM
from turngpt.turngpt_utils import (
    get_positive_and_negative_indices,
    batch_to_context_ablation_batch,
    turns_to_turngpt_tensors,
    get_focus_indices,
    input_ids_to_token, # DH
    get_focus_n_tokens, # DH
    get_false_tokens, # DH
    find_turn_with_index, # DH
    get_turns,
    find_turn_context,
    save_txt, # DH
)

import matplotlib.pyplot as plt


def load():
    parser = ArgumentParser()
    parser.add_argument("--chunk_size", default=512, type=int)
    parser = TurnGPTEval.add_eval_specific_args(parser)

    # Data
    parser = TurnGPTDM.add_data_specific_args(parser)
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["coached"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",  # val, test, or all
    )
    temp_args, _ = parser.parse_known_args()

    # Add all datasets
    datasets = temp_args.datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()

    # Tokenizer
    tokenizer = torch.load(args.tokenizer)

    # Data
    dm = TurnGPTDM(args, tokenizer)
    dm.prepare_data()

    # Choose Model
    if args.model == "mini":
        from turngpt.models.gpt_mini import TurnGPTMini

        model = TurnGPTMini.load_from_checkpoint(checkpoint_path=args.checkpoint)
    elif args.model == "pretrained":
        from turngpt.models.pretrained import TurnGPTPretrained

        model = TurnGPTPretrained.load_from_checkpoint(checkpoint_path=args.checkpoint)

    return model, dm, args


def get_dataloader(dm, args):
    """ We always use the dm.test_dataloader() -> so we change the filepaths to the relevant splits """
    dm.setup("fit")
    dm.setup("test")
    if args.split == "train":
        dm.test_dset.filepaths = dm.train_dset.filepaths
    elif args.split == "val":
        dm.test_dset.filepaths = dm.val_dset.filepaths
    elif args.split == "test":
        # No path fix required
        pass
    else:  # all
        dm.test_dset.filepaths += dm.train_dset.filepaths
        dm.test_dset.filepaths += dm.val_dset.filepaths
    return dm.test_dataloader()


class TurnGPTEval(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.sp1_idx = tokenizer.convert_tokens_to_ids("<speaker1>")
        self.sp2_idx = tokenizer.convert_tokens_to_ids("<speaker2>")
        self.pad_idx = tokenizer.pad_token_id
        print("sp1_idx: ", self.sp1_idx)
        print("sp2_idx: ", self.sp2_idx)
        print("pad_idx: ", self.pad_idx)

    def trp(self, input_ids, speaker_ids, output_attentions=False):
        # NOTE: The 'output_attentions' parameter was changed in transformers 3.x. Plz lock down to transformers version 2.5.1
        out = self.model(
            input_ids, speaker_ids=speaker_ids, output_attentions=output_attentions    # DH: problem
        )
        logits = out["logits"]
        prob = F.softmax(logits, dim=-1)
        # DH: get sp1_idx and sp2_idx values for each token
        prob = torch.stack((prob[..., self.sp1_idx], prob[..., self.sp2_idx]), dim=-1)
        # DH: get the larger one between sp1_idx prob and sp2_idx prob as trp
        prob, _ = prob.max(dim=-1)
        ret = {"trp": prob}
        if output_attentions:
            ret["attention"] = out["attention"]
        return ret

    def cross_entropy(self, dloader):
        self.eval()

        with torch.no_grad():
            total_loss = []
            for b in tqdm(dloader, desc="perplexity"):
                o = self.model.validation_step(
                    [b[0].to(self.device), b[1].to(self.device)]
                )
                total_loss.append(o["val_loss"])
            total_loss = torch.stack(total_loss)
            loss_sd = total_loss.std().item()
            loss = total_loss.mean().item()

        ppl = math.exp(loss)
        return loss, ppl

    @torch.no_grad()
    def classification(self, test_dataloader):
        self.eval()
        all_pos = []
        all_neg = []
        for batch in tqdm(test_dataloader, desc="Classification"):
            input_ids, speaker_ids = batch[0], batch[1]
            # label = input_ids[:, 1:]
            input_ids = input_ids[:, :-1]
            speaker_ids = speaker_ids[:, :-1]
            # get the TRP probabilities
            out = self.trp(input_ids.to(self.device), speaker_ids.to(self.device))
            trp = out["trp"]
            # Find positive/negative labels
            ts_inds, non_ts_inds = get_positive_and_negative_indices(
                input_ids, self.sp1_idx, self.sp2_idx, self.pad_idx
            )
            # extract positive/negative-guesses
            all_pos.append(trp[ts_inds].cpu())
            all_neg.append(trp[non_ts_inds].cpu())
        all_pos = torch.cat(all_pos)
        all_neg = torch.cat(all_neg)

        score = fscores(all_pos, all_neg)
        best_bacc, i = score["bacc"].max(dim=0)
        best_thresh = score["cutoffs"][i]
        return {
            "bacc": best_bacc,
            "thresh": best_thresh,
            "all_baccs": score["bacc"],
            "all_thresh": score["cutoffs"],
            "positive_guesses": all_pos,
            "negative_guesses": all_neg,
        }

    @torch.no_grad()
    def context_ablation(
        self,
        test_dataloader,
        n_context=4,
        omit_speaker_toks_as_negatives=False,
        max_batch_size=20,
        max_batches=None,
    ):
        self.eval()

        pos_guesses = {context: [] for context in range(n_context + 1)}
        neg_guesses = {context: [] for context in range(n_context + 1)}

        # batch = next(iter(test_dataloader))
        for n_batch, batch in enumerate(
            tqdm(test_dataloader, total=max_batches, desc="Context Ablation")
        ):
            if max_batches is not None and n_batch == max_batches:
                break
            input_ids, speaker_ids = batch[0], batch[1]

            context_batch_data = batch_to_context_ablation_batch(
                input_ids,
                speaker_ids,
                n_context=n_context,
                sp1_idx=self.sp1_idx,
                sp2_idx=self.sp2_idx,
                omit_speaker_toks_as_negatives=omit_speaker_toks_as_negatives,
                sort=True,  # sort the batch by lengths for more efficent forward pass
            )

            context_ids = context_batch_data["context_ids"]  # new input_ids
            context_speaker = context_batch_data["context_speaker"]  # new speaker_ids
            pos_context = context_batch_data[
                "pos_context"
            ]  # the given context for the positive labels
            neg_context = context_batch_data[
                "neg_context"
            ]  # the given context for the negative labels
            pos_indices = context_batch_data[
                "pos_indices"
            ]  # indices for positive guesses
            neg_indices = context_batch_data[
                "neg_indices"
            ]  # indices for negative guesses

            # Iterate over the context datapoints so the batch size is fixed (easily overextend the gpu memory)
            all_trps = []
            for i in range(0, len(context_ids), max_batch_size):
                end = i + max_batch_size

                # pad the tensors for model input
                tmp_context_ids = pad_sequence(
                    context_ids[i:end], padding_value=self.pad_idx, batch_first=True
                )
                tmp_context_speaker = pad_sequence(
                    context_speaker[i:end], padding_value=self.pad_idx, batch_first=True
                )

                # the context given for each datapoint
                tmp_pos_context = torch.tensor(pos_context[i:end])
                tmp_neg_context = torch.cat(neg_context[i:end])

                # Positive guess indices
                tmp_pos_inds = torch.stack(pos_indices[i:end])
                tmp_pos_batch = torch.arange(len(tmp_pos_inds))

                # Negative guess indices
                tmp_neg_inds = neg_indices[i:end]
                tmp_neg_batch = torch.cat(
                    [torch.tensor(len(f) * [i]) for i, f in enumerate(tmp_neg_inds)]
                )
                tmp_neg_inds = torch.cat(tmp_neg_inds)

                # __import__("ipdb").set_trace()

                # FORWARD PASS
                trp = self.trp(
                    tmp_context_ids.to(self.device),
                    tmp_context_speaker.to(self.device),
                )["trp"].cpu()

                # Positive/Negative predictions
                pos_prediction = trp[(tmp_pos_batch, tmp_pos_inds)]  # positive guesses
                neg_prediction = trp[(tmp_neg_batch, tmp_neg_inds)]  # negative guesses

                # Map back to relevant context lens
                for i in range(n_context + 1):
                    p = pos_prediction[tmp_pos_context == i]
                    if len(p) > 0:
                        pos_guesses[i].append(p)
                    n = neg_prediction[tmp_neg_context == i]
                    if len(n) > 0:
                        neg_guesses[i].append(n)

        # single tensor for each context lenght
        for k, v in pos_guesses.items():
            pos_guesses[k] = torch.cat(v)
        for k, v in neg_guesses.items():
            neg_guesses[k] = torch.cat(v)
        return {"positive": pos_guesses, "negative": neg_guesses}

    @torch.no_grad()
    def context_attention(
        self,
        test_dataloader,
        prob_thresh=0.2,
        n_context=4,
        omit_speaker_token=True,
        normalize=True,
    ):
        self.model.eval()
        turn_context_attention = []
        skipped = 0  # n_batches skipped
        for batch in tqdm(test_dataloader, desc="Context Attention"):
            input_ids, speaker_ids = batch[0], batch[1]

            # Get likelihood over trp / turn-shifts over btch
            trp = self.trp(
                input_ids.to(self.device),
                speaker_ids.to(self.device),
                output_attentions=True,
            )
            attention = trp["attention"]  # (B, layers, heads, N, N)

            # Get the points where the model assigned a larger trp likelihood > 'prob_thresh'
            # with at least 'n_context' previous turns (history/context)
            focus_bs, focus_inds = get_focus_indices(
                trp["trp"],
                input_ids,
                prob_thresh=prob_thresh,
                n_context=n_context,
                sp1_idx=self.sp1_idx,
                sp2_idx=self.sp2_idx,
            )

            # Skip batch if no suitable targets was found
            if len(focus_bs) == 0:
                skipped += 1
                continue

            # get all turns in batch
            turns = get_turns(input_ids, self.sp1_idx, self.sp2_idx)

            # Iterate over all the valid focus points and extract the attention over the context and current turn
            for i, b in enumerate(focus_bs):
                focus_index = focus_inds[i]
                tmp_turn_context = find_turn_context(focus_index, turns[b], n_context)

                # Iterate over all context (and current) turn and extract attention
                tmp_context_att = []
                for t in tmp_turn_context:
                    if omit_speaker_token:
                        tmp_context_att.append(
                            attention[b, :, :, focus_index, t[0] + 1 : t[1]].sum()
                        )
                    else:
                        tmp_context_att.append(
                            attention[b, :, :, focus_index, t[0] : t[1]].sum()
                        )
                tmp_context_att = torch.stack(tmp_context_att).cpu()

                # Normalizes the attention values
                if normalize:
                    tmp_context_att /= tmp_context_att.sum()

                turn_context_attention.append(tmp_context_att)

        turn_context_attention = torch.stack(turn_context_attention)
        print("Context attention samples: ", turn_context_attention.shape[0])
        print("Skipped batches: ", skipped)
        return turn_context_attention

    def integrated_gradient(
        self,
        input_ids,
        speaker_ids,
        focus_index,
        focus_token,
        m=20,
        baseline_idx=50256,
        use_pbar=False,
    ):
        """
        Calculate the Integrated Gradient from the paper:
            "Axiomatic Attribution for Deep Networks", Sundararajan et al. https://arxiv.org/abs/1703.01365

        Based on SOURCE: https://github.com/ankurtaly/Integrated-Gradients

        Arguments:

            :param input_ids:       torch.tensor, token indices input
            :param speaker_ids:     torch.tensor, speaker indices input
            :param focus_index:     int, target index (time) to calculate IG from
            :param focus_token:     int, taget (token) to calculate IG from
            :param m:               int, number of interpolation steps
            :param baseline_idx:    int, what token index to use as baseline

        Returns:
            dict,                   {'ig', 'all_predictions', 'focus_prob', 'error'}
        """
        self.eval()

        # Device
        input_ids = input_ids.to(self.device)
        speaker_ids = speaker_ids.to(self.device)

        # Get input embeddings
        # wte: word token embeddings; wpe: word position embeddings
        input_embeds = self.model.model.model.transformer.wte(input_ids)
        #print(">> input_embeds", input_embeds, input_embeds.size())
        #input(">> press any key...")

        # Baseline embeddings
        # Speaker indices are kept intact. The turns are fixed and we care about the gradient of the words
        baseline_ids = torch.ones_like(input_ids).fill_(baseline_idx)
        baseline_ids[
            input_ids == self.sp1_idx
        ] = self.sp1_idx  # speaker indices are kept
        baseline_ids[
            input_ids == self.sp2_idx
        ] = self.sp2_idx  # speaker indices are kept
        baseline = self.model.model.model.transformer.wte(
            baseline_ids
        )  # get baseline embeddings
        #print(">> baseline", baseline, baseline.size())
        #input(">> press any key...")

        # Create the linear interpolation inputs, between actual input and baseline, for the IG algorithm
        interpolated_inputs = []
        with torch.no_grad():
            diff_vector = input_embeds - baseline
            for k in range(m):  # omit m+1 and add the actual input at the end
                interpolated_inputs.append(baseline + float(k) / m * diff_vector)
        interpolated_inputs.append(input_embeds.detach())
        #print(">> interpolated_inputs", interpolated_inputs, len(interpolated_inputs))
        #input(">> press any key...")

        # Used to zero gradients
        optim = torch.optim.SGD(self.parameters(), lr=1)
        # Or
        # def zero_grad(model):
        #     """Clears the gradients of all optimized :class:`torch.Tensor` s."""
        #     for p in model.parameters:
        #         if p.grad is not None:
        #             p.grad.detach_()
        #             p.grad.zero_()

        if use_pbar:
            pbar = tqdm(interpolated_inputs, desc="IG")
        else:
            pbar = interpolated_inputs

        # Iterate over the scaled inputs and calculate the gradient.
        grads = []
        predictions = []
        for tmp_input_embeds in pbar:
            tmp_input_embeds = tmp_input_embeds.to(self.device)
            tmp_input_embeds.requires_grad = True
            # out = self.model.model.transformer_forward(tmp_input_embeds, speaker_ids)
            out = self.model.model.body_from_embedding(tmp_input_embeds, speaker_ids)
            lm_logits = self.model.model.model.lm_head(out["z"])
            probs = F.softmax(lm_logits, dim=-1)
            probs[:, focus_index, focus_token].backward()
            grads.append(tmp_input_embeds.grad.cpu())
            predictions.append(probs.detach().cpu())
            optim.zero_grad()
        grads = torch.cat(grads)
        predictions = torch.cat(predictions)
        #print(">> grads", grads, grads.size())
        #print(">> predictions", predictions, predictions.size())
        #input(">> press any key...")

        # Use trapezoidal rule to approximate the integral.
        # See Section 4 of the following paper for an accuracy comparison between
        # left, right, and trapezoidal IG approximations:
        # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
        # https://arxiv.org/abs/1908.06214
        with torch.no_grad():
            grads = (grads[:-1] + grads[1:]) / 2.0
            avg_grads = grads.mean(dim=0)
            #print(">> new grad:", grads, grads.size())
            #print(">> avg_grads:", avg_grads, avg_grads.size())
            #input(">> press any key...")
            integrated_gradients = diff_vector.cpu() * avg_grads  # shape: <inp.shape>
            #print(">> integrated_gradients:", integrated_gradients, integrated_gradients.size())
            #input(">> press any key...")

        # Check computation
        # ------------------------------------------------------------------------------------------------
        # "In practice, we find that somewhere between 20 and 300 steps are enough to approximate the
        # integral (within 5%); we recommend that developers check that the attributions approximately adds
        # up to the difference beween the score at the input and that at the baseline
        # (cf. Proposition 1), and if not increase the step-size m." - End of Section 5: Computing Integrated Gradients
        # ------------------------------------------------------------------------------------------------
        with torch.no_grad():
            score_diff = (
                predictions[-1, focus_index, focus_token]
                - predictions[0, focus_index, focus_token]
            ).item()
            # print(integrated_gradients.shape, predictions.shape)
            diff = integrated_gradients.sum().item() - score_diff
            error_perc = abs(round(diff * 100 / score_diff, 3))

        if use_pbar:
            #print(f"Error: {error_perc}%")
            if error_perc >= 5:
                print("Error is larger than 5%. Increase 'm'...")
        return {
            "ig": integrated_gradients,
            "focus_prob": predictions[-1, focus_index, focus_token],
            "all_predictions": predictions,
            "error": error_perc,
        }

    def context_IG(
        self, test_dataloader, prob_thresh=0.2, n_context=4, m=70, normalize=True
    ):
        print("Calculating the IG for all valid turn-shift predictions")
        print("This function is very slow (forward/backward pass for each target focus)")

        turn_context_ig = []
        batch_skipped = 0  # n_batches skipped
        error_skipped = 0  # skipped due to IG calculation was over recommended error

        #print(">> test_dataloader:",[item for item in test_dataloader])
        #input(">> press any key to continue")
        for batch in tqdm(test_dataloader, desc="Context IG"):
            #print(">> batch:", batch, len(batch))
            #input(">> press any key...")
            input_ids, speaker_ids = batch[0], batch[1]
            #print(">> input_ids", input_ids, input_ids.size() )
            #print(">> speaker_ids", speaker_ids, speaker_ids.size())
            #input(">> press any key...")

            # Get likelihood over trp / turn-shifts over batch 
            # (DH: tensors are copied to GPU device, and computations are done there)
            trp = self.trp(input_ids.to(self.device), speaker_ids.to(self.device))
            #print(">> trp:", trp['trp'], trp['trp'].size())
            #input(">> press any key...")

            # Get the points where the model assigned a larger trp likelihood > 'prob_thresh'
            # with at least 'n_context' previous turns (history/context)
            focus_bs, focus_inds = get_focus_indices(
                trp["trp"],
                input_ids,
                prob_thresh=prob_thresh,
                n_context=n_context,
                sp1_idx=self.sp1_idx,
                sp2_idx=self.sp2_idx,
            )
            #print(">> focus_bs", focus_bs, len(focus_bs) )
            #print(">> focus_inds", focus_inds, len(focus_inds) )
            #input(">> press any key...")

            # Skip batch if no suitable targets was found
            if len(focus_bs) == 0:
                batch_skipped += 1
                continue

            # get all turns in batch
            turns = get_turns(input_ids, self.sp1_idx, self.sp2_idx)
            #print(">> turns:", turns, [turn.size() for turn in turns] )
            #input(">> press any key...")

            # Iterate over all the valid focus points and extract the attention over the context and current turn
            for i, b in enumerate(focus_bs):
                #print(">> first item of focus_bs:", i, b)
                #input(">> press any key...")
                focus_index = focus_inds[i]
                tmp_turn_context = find_turn_context(focus_index, turns[b], n_context)
                #print(">> focus_index:", focus_index)
                #print(">> turns[b]:", turns[b])
                #print(">> tmp_turn_context:", tmp_turn_context)
                #input(">> press any key...")

                # Only the past is relevant for the gradient computation
                tmp_input = input_ids[b, : focus_index + 1]
                tmp_speaker = speaker_ids[b, : focus_index + 1]
                # print(">> tmp_input:", tmp_input, tmp_input.size())
                # print(">> tmp_speaker:", tmp_speaker, tmp_speaker.size())
                # input()

                # the relevant focus token is the opposite of the speaker at focus_index
                # corresponds to shifting turn (prediction after the last word is another <speaker>)
                focus_token = (
                    self.sp1_idx
                    if tmp_speaker[focus_index] == self.sp2_idx
                    else self.sp2_idx
                )
                #print(">> focus_token", focus_token)

                # Using a try statement here because this whole function is so slow
                # so we might want to interrupt it but still get some values back
                try:
                    # ig.keys:  ['ig', 'focus_prob', 'all_predictions', 'error']
                    # ig['ig']: (B, N, hidden_dim) e.g (1, 19, 768)
                    ig = self.integrated_gradient(
                        tmp_input.unsqueeze(0),  # unsqueeze batch dim
                        tmp_speaker.unsqueeze(0),  # unsqueeze batch dim
                        focus_index=focus_index,
                        focus_token=focus_token,
                        m=m,
                        baseline_idx=self.pad_idx,
                        use_pbar=True,  # DH add pbar
                    )
                except KeyboardInterrupt:
                    return torch.stack(turn_context_ig)
                print(">> ig.ig", ig['ig'])
                print(">> ig.focus_prob", ig['focus_prob'])
                #print(">> ig.all_predictions", ig['all_predictions'])
                print(">> ig.error", ig['error'])

                # Skip IG calculation with error larger than 5% which is recommended in the paper
                if ig["error"] >= 5:
                    error_skipped += 1
                    continue

                # Iterate over all context (and current) turn and extract IG-sum for each turn
                tmp_context_ig = []
                for t in tmp_turn_context:
                    # Always omit speaker-tokens (they will have 0 IG by definition anyways)
                    tmp_context_ig.append(ig["ig"][0, t[0] + 1 : t[1]].sum())
                #print(">> tmp_context_ig before", tmp_context_ig)
                tmp_context_ig = torch.stack(tmp_context_ig).cpu()
                print(">> tmp_context_ig", tmp_context_ig)

                # Normalizes the IG values by the output probability of focus_index
                # The gradient contribution should add up to 'focus_prob'
                if normalize:
                    tmp_context_ig /= ig["focus_prob"]
                turn_context_ig.append(tmp_context_ig)
                print(">> turn_context_ig", turn_context_ig)

        turn_context_ig = torch.stack(turn_context_ig)
        print("Context attention samples: ", turn_context_ig.shape[0])
        print("Skipped batches: ", batch_skipped)
        print("Skipped error: ", error_skipped)
        return turn_context_ig

    # DH: add
    def focus_word_IG(
        self, data_list, m=70, normalize=True
    ):
        print("Calculating the IG for turn-shift predictions of specific word in the turns")
        print("This function is very slow (forward/backward pass for each target focus)")

        turns_word = []
        turns_word_ig = []
        batch_skipped = 0  # n_batches skipped
        error_skipped = 0  # skipped due to IG calculation was over recommended error

        #print(">> data_list:",[item for item in data_list])
        #input(">> press any key to continue")
        for batch in tqdm(data_list, desc="Word IG"):
            input_ids, speaker_ids, focus_id = batch[0], batch[1], batch[2]

            focus_bs, focus_inds = get_focus_n_tokens(
                input_ids, torch.Tensor([focus_id]), 
            )
            # print(">> focus_bs", focus_bs, len(focus_bs) )
            # print(">> focus_inds", focus_inds, len(focus_inds) )
            # input(">> press any key...")

            # Skip batch if no suitable targets was found (i.e. no enough words prior to focus point)
            if len(focus_bs) == 0:
                batch_skipped += 1
                continue

            # Extract the attention over the words of the focus point
            for i, b in enumerate(focus_bs):
                # i, b: e.g. 0(index), 0(batch_num)
                focus_index = focus_inds[i]
                # focus_index: e.g. 15

                # Only the past is relevant for the gradient computation
                tmp_input = input_ids[b, : focus_index+1]
                tmp_speaker = speaker_ids[b, : focus_index+1]
                #print("tmp", tmp_input, tmp_speaker)
                #input(">> press any key...")
                # tmp_input: e.g. tensor([ 345, 1826, 757, 50257, 9439])
                # tmp_speaker: e.g. tensor([ 50258, 50258, 50258, 50257, 50257])

                # the relevant focus token is the opposite of the speaker at focus_index
                # corresponds to shifting turn (prediction after the last word is another <speaker>)
                focus_token = (
                    self.sp1_idx
                    if tmp_speaker[-1] == self.sp2_idx
                    else self.sp2_idx
                )

                # Using a try statement here because this whole function is so slow
                # so we might want to interrupt it but still get some values back
                try:
                    # ig.keys:  ['ig', 'focus_prob', 'all_predictions', 'error']
                    # ig['ig']: (B, N, hidden_dim) e.g. (1, 19, 768)
                    ig = self.integrated_gradient(
                        tmp_input.unsqueeze(0),  # unsqueeze batch dim
                        tmp_speaker.unsqueeze(0),  # unsqueeze batch dim
                        focus_index=focus_index,
                        focus_token=focus_token,
                        m=m,
                        baseline_idx=self.pad_idx,
                        use_pbar=True,  # DH add pbar
                    )
                except KeyboardInterrupt:
                    return torch.stack(turns_word_ig)

                # Skip IG calculation with error larger than 5% which is recommended in the paper
                if ig["error"] >= 5:
                    print(f"Warning: IG error {ig['error']}%")
                    error_skipped += 1
                    continue

                # Iterate over all context (and current) turn and extract IG-sum for each turn
                tmp_context_ig = []
                for tok in ig["ig"][0]: # ig is always of size [1, N, hidden_dim] (one batch one iteration)
                    tmp_context_ig.append(tok.sum())
                tmp_context_ig = torch.stack(tmp_context_ig).cpu()
                # print(">> tmp_context_ig", tmp_context_ig)

                # Normalizes the IG values by the output probability of focus_index
                # The gradient contribution should add up to 'focus_prob'
                if normalize:
                    tmp_context_ig /= ig["focus_prob"]
                # keep only n_token tokens prior to and including the focus token
                #start_indx = -n_token if n_token<len(tmp_context_ig) else 0
                turns_word_ig.append( tmp_context_ig )
                turns_word.append( tmp_input )
                # print(">> turn_context_ig", turns_word_ig)

        print("Skipped error: ", error_skipped)
        return turns_word_ig, turns_word
    
    # DH: add
    def word_IG(
        self, test_dataloader, prob_thresh=0.2, m=70, normalize=True, actual_end=False, 
        save_step=10, tokenizer=None, savepath="", restore_from=0, end_step=None, 
    ):
        print("Calculating the IG for all valid turn-shift predictions")
        print("This function is very slow (forward/backward pass for each target focus)")

        turns_word = []
        false_word_ig = []
        batch_skipped = 0  # n_batches skipped
        error_skipped = 0  # skipped due to IG calculation was over recommended error
        not_in_turn_skipped = 0 # DH: skipped due to false token not found in turns

        step = 0
        for batch in tqdm(test_dataloader, desc=f"{actual_end} Word IG"):

            # skip older data
            if step >= restore_from:

                input_ids, speaker_ids = batch[0], batch[1]
                #print(">> input_ids", input_ids, input_ids.size() )
                #print(">> speaker_ids", speaker_ids, speaker_ids.size())

                # Get likelihood over trp / turn-shifts over batch 
                # (DH: tensors are copied to GPU device, and computations are done there)
                trp = self.trp(input_ids.to(self.device), speaker_ids.to(self.device))

                if not actual_end:
                    # Get the points where the model makes incorrect predictions between trp & 'prob_thresh'
                    focus_bs, focus_inds = get_false_tokens(
                        trp["trp"], 
                        input_ids, 
                        prob_thresh=prob_thresh, 
                        sp1_idx=self.sp1_idx,
                        sp2_idx=self.sp2_idx
                    )
                else:
                    # Get the points where the model assigned a larger trp likelihood > 'prob_thresh'
                    focus_bs, focus_inds = get_focus_indices(
                        trp["trp"],
                        input_ids,
                        prob_thresh=prob_thresh,
                        n_context=0,
                        sp1_idx=self.sp1_idx,
                        sp2_idx=self.sp2_idx,
                    )

                # get all turns in batch
                turns = get_turns(input_ids, self.sp1_idx, self.sp2_idx)

                # Skip batch if no suitable targets was found
                if len(focus_bs) == 0:
                    batch_skipped += 1
                    continue

                # Extract the attention over the words of the focus point
                for i, b in enumerate(tqdm(focus_bs, desc=f"Batch{step}")):
                    # i, b: e.g. 0(index), 0(batch_num)
                    false_index = focus_inds[i]
                    # false_index: e.g. 8
                    try:
                        t_s, t_e = find_turn_with_index(false_index, turns[b])
                    except:
                        not_in_turn_skipped += 1
                        continue

                    if t_s==-1:
                        not_in_turn_skipped += 1
                        continue

                    # Only the past is relevant for the gradient computation
                    tmp_input = input_ids[b, t_s : false_index+1]
                    tmp_speaker = speaker_ids[b, t_s : false_index+1]
                    # tmp_input: e.g. tensor([50257,   523,   345,   821,  1016,   510])
                    # tmp_speaker: e.g. tensor([50257, 50257, 50257, 50257, 50257, 50257])

                    # the relevant focus token is the opposite of the speaker at focus_index
                    # corresponds to shifting turn (prediction after the last word is another <speaker>)
                    focus_token = (
                        self.sp1_idx
                        if tmp_speaker[false_index-t_s] == self.sp2_idx
                        else self.sp2_idx
                    )
                    #print(">> focus_token", focus_token)

                    # Using a try statement here because this whole function is so slow
                    # so we might want to interrupt it but still get some values back
                    try:
                        # ig.keys:  ['ig', 'focus_prob', 'all_predictions', 'error']
                        # ig['ig']: (B, N, hidden_dim) e.g. (1, 19, 768)
                        ig = self.integrated_gradient(
                            tmp_input.unsqueeze(0),  # unsqueeze batch dim
                            tmp_speaker.unsqueeze(0),  # unsqueeze batch dim
                            focus_index=false_index-t_s,
                            focus_token=focus_token,
                            m=m,
                            baseline_idx=self.pad_idx,
                            #use_pbar=True,  # DH add pbar
                        )
                    except KeyboardInterrupt:
                        return torch.stack(false_word_ig)
                    #print(">> ig.ig", ig['ig'])
                    #print(">> ig.focus_prob", ig['focus_prob'])

                    # Skip IG calculation with error larger than 5% which is recommended in the paper
                    if ig["error"] >= 5:
                        print(">> large IG error: ", ig['error'])
                        error_skipped += 1
                        continue

                    # Iterate over all context (and current) turn and extract IG-sum for each turn
                    tmp_context_ig = []
                    for tok in ig["ig"][0]: # ig is always of size [1, N, hidden_dim] (one batch one iteration)
                        #print(tok)
                        tmp_context_ig.append(tok.sum())
                    tmp_context_ig = torch.stack(tmp_context_ig).cpu()
                    #print(">> tmp_context_ig", tmp_context_ig)

                    # Normalizes the IG values by the output probability of focus_index
                    # The gradient contribution should add up to 'focus_prob'
                    if normalize:
                        tmp_context_ig /= ig["focus_prob"]
                    # keep only n_token tokens prior to and including the focus token
                    false_word_ig.append( tmp_context_ig )
                    turns_word.append( tmp_input )
                    #print(">> false_word_ig", false_word_ig)

            step += 1
            # save partial results if it is new checkpoints
            if (not step % save_step) and (step >= restore_from):
                save_txt(false_word_ig, turns_word, tokenizer, 
                        join(savepath, f"exp_ig/word_ig_{actual_end}{step//save_step}.txt"), 
                        skipped_dic={
                            "skipped_batches": batch_skipped, 
                            "skipped_error": error_skipped, 
                            "skipped_not_in_turn": not_in_turn_skipped, 
                        }
                )
                # empty two lists
                false_word_ig = []
                turns_word = []
                # reset skipped counters
                batch_skipped = 0
                error_skipped = 0
                not_in_turn_skipped = 0
            # quit if reaching the end_step
            if end_step and step >= end_step:
                break

        return false_word_ig, turns_word

    def get_trp(self, input_ids, speaker_ids):
        out = self.trp(input_ids.to(self.device), speaker_ids.to(self.device))
        return out["trp"]

    def data_statistic(self, dataloader, tokenizer):
        id_dict = {}
        for batch in tqdm(dataloader, desc="Statistic"):
            input_ids, speaker_ids = batch[0], batch[1]
            # input_ids: [[...], [...]] shape[0]=batch_size, shape[1]=chunck_size
            for bat_input in input_ids:
                for iid in bat_input:
                    if iid.item() in (self.sp1_idx, self.sp2_idx, 50256):
                        continue
                    else:
                        try:
                            id_dict[iid.item()] += 1
                        except:
                            id_dict[iid.item()] = 1
        
        # decode to tokens
        token_dict = {}
        s = sum(id_dict.values())
        for key, value in id_dict.items():
            token = tokenizer.decode(key)
            token_dict[token] = value/s

        return sorted(token_dict.items(), key=lambda x:x[1], reverse=True)

    def sample_data(self, dataloader, tokenizer, v=500):
        from random import sample

        sample_list = []
        for batch in tqdm(dataloader, desc="Sampling"):
            input_ids, speaker_ids = batch[0], batch[1]
            # input_ids: [[...], [...]] shape[0]=batch_size, shape[1]=chunck_size
            for bat_input in input_ids:
                tmp_list = []
                for iid in bat_input:
                    if iid.item() in (self.sp1_idx, self.sp2_idx, 50256):
                        continue
                    else:
                        tmp_list.append(tokenizer.decode(iid.item()))
                sample_list.append(tmp_list)
        
        # sample choice
        return sample(sample_list, min(v, len(sample_list)))

    @torch.no_grad()
    def prediction_histogram(
        self,
        input_ids,
        speaker_ids,
        tokenizer,
        n_samples=1000,
        batch_size=50,
        horizon=50,
        start_after_first_turn=False,
    ):
        assert n_samples % batch_size == 0, "please make n_sample % batch_size = 0"
        # First turn is used as context
        tokens = convert_ids_to_tokens(input_ids, tokenizer)[0]
        turns = get_turns(input_ids, self.sp1_idx, self.sp2_idx)

        # start sampling after the first turn
        max_index = len(input_ids[0]) - 1
        if start_after_first_turn:
            start_index = turns[0][1, 0]
        else:
            start_index = 0

        # input_ids = input_ids.to(model.device)
        # speaker_ids = speaker_ids.to(model.device)

        prediction_dist = []
        start_words = []
        samples = []
        for step in tqdm(range(max_index - start_index), desc="Prediction Hist"):
            current_ind = start_index + step
            tmp_in = input_ids[
                :, : current_ind + 1
            ]  # range must include current ind (i.e. +1)
            tmp_sp = speaker_ids[
                :, : current_ind + 1
            ]  # range must include current ind (i.e. +1)
            start_words.append(tokens[current_ind])
            tmp_hist = []
            for n in range(n_samples // batch_size):
                sample_out = self.model.sample(
                    tmp_in,
                    tmp_sp,
                    sp1_idx=self.sp1_idx,
                    sp2_idx=self.sp2_idx,
                    batch_size=batch_size,
                    steps=horizon,
                    top_k=3,
                    temperature=1.0,
                    sample=True,
                    stop_at_turn_shift=True,
                    use_pbar=False,
                )
                tmp_hist += sample_out["k"]  # sample_out['k'] is a list
                if n == 0:
                    if isinstance(sample_out["output_ids"], list):
                        # append first sample-output in list
                        samples.append(
                            convert_ids_to_tokens(
                                sample_out["output_ids"][0][
                                    current_ind + 1 :
                                ].unsqueeze(0),
                                tokenizer,
                            )[0]
                        )
                    else:
                        # append first batch in tensor sample-output
                        samples.append(
                            convert_ids_to_tokens(
                                sample_out["output_ids"][:1, current_ind + 1 :],
                                tokenizer,  # first batch only
                            )[0]
                        )
            prediction_dist.append(tmp_hist)
        return {
            "prediction_distribution": prediction_dist,
            "start_words": start_words,
            "samples": samples,
        }

    @staticmethod
    def add_eval_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--checkpoint", default=None, type=str)
        parser.add_argument("--tokenizer", default=None, type=str)
        parser.add_argument(
            "--plot",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--model",
            type=str,
            default="pretrained",  # mini
        )
        parser.add_argument(
            "--perplexity",
            action="store_false",   # NOTE: use --perplexity will cancel it!
            default=True,
        )
        parser.add_argument(
            "--classification",
            action="store_false",   # NOTE: use --classification will cancel it!
            default=True,
        )
        parser.add_argument(
            "--context_ablation",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--context_attention",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--context_ig",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--prediction_hist",
            action="store_true",
            default=False,
        )
        parser.add_argument(    # added by DH
            "--word_ig",
            action="store_true",
            default=False,
        )
        parser.add_argument(    # added by DH
            "--false_word_ig",
            action="store_true",
            default=False,
        )
        parser.add_argument(    # added by DH
            "--true_word_ig",
            action="store_true",
            default=False,
        )
        parser.add_argument(    # added by DH
            "--trp_sample",
            action="store_true",
            default=False,
        )
        parser.add_argument(    # added by DH
            "--data_stat",
            action="store_true",
            default=False,
        )
        return parser


class Plots:
    @staticmethod
    def trp_sample(
        trp,
        tokens,
        text_size=10,
        xy_label_size=10,
        style="seaborn-darkgrid",
        plot=False,
        highlightx=None,
    ):
        plt.style.use(style)
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        x = torch.arange(len(trp))
        ax.bar(x, trp)
        ax.set_ylim([0, 1])
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(
            tokens, fontdict={"fontsize": text_size}, rotation=90   # , "fontweight": "bold"
        )
        ax.set_ylabel("TRP Prediction", fontdict={"fontsize": xy_label_size})
        if highlightx:
            ax.get_xticklabels()[tokens.index(highlightx)].set_color("blue")
        
        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def bacc(classification_score, style="seaborn-darkgrid", plot=False):
        plt.style.use(style)
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(classification_score["all_thresh"], classification_score["all_baccs"])
        ax.set_ylim([0.5, 1])
        ax.set_xlabel("probability thresh")
        ax.set_ylabel("bAcc")
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def context_ablation(
        score,
        ax=None,
        color="b",
        linestyle="--",
        style="seaborn-darkgrid",
        plot=False,
    ):
        plt.style.use(style)

        # if ax is not given create a new fig
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = None

        ax.plot(
            torch.arange(len(score)), score, color=color, linestyle=linestyle, alpha=0.6
        )
        ax.scatter(
            torch.arange(len(score)),
            score,
            marker=".",
            linewidth=3,
            color=color,
            alpha=0.6,
        )
        ax.set_xlabel("context")
        ax.set_ylim([0.6, 1])
        ax.set_ylabel("bAcc")
        ax.set_xticks(range(len(score)))
        ax.set_xticklabels(list(range(len(score))))
        ax.set_xlabel("Context Turns")
        plt.gca().invert_xaxis()
        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def context_attention(
        context_attention,
        notch=False,
        showfliers=False,
        showmeans=True,
        meanline=True,
        whis=None,
        linewidth=0.5,
        style="seaborn-darkgrid",
        flierprops=None,
        figsize=(7, 6),
        ylabel="Attention",
        label_size=15,
        tick_size=None,
        ylim=[-0.05, 1],
        plot=False,
    ):
        plt.style.use(style)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        bps = []
        for t in range(context_attention.shape[-1]):
            tmp = ax.boxplot(
                context_attention.t(),
                notch=notch,
                whis=whis,
                showmeans=showmeans,
                meanline=meanline,
                showfliers=showfliers,
                boxprops={"linewidth": linewidth},
                capprops={"linewidth": linewidth},
                whiskerprops={"linewidth": linewidth},
                medianprops={"linewidth": linewidth},
                meanprops={"linewidth": linewidth},
                flierprops=flierprops,
            )
            bps.append(tmp)
        ax.grid("on")
        ax.axhline(y=0, color="k", alpha=0.2, linewidth=0.9)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=label_size)
        ax.set_xlabel("Context Turns", fontsize=label_size)
        ax.set_xticklabels(ax.get_xticks()[::-1] - 1)
        ax.tick_params(axis="y", labelsize=tick_size)
        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def integrated_gradient(ig, tokens, focus, reduction=None, plot=False):
        plt.style.use("seaborn-darkgrid")
        fig, ax = plt.subplots(1, 1, figsize=(4,3))
        if reduction == "sum":
            i_g = ig.sum(dim=-1)
            ax.bar(torch.arange(len(tokens)), i_g[0])
        elif reduction == "mean":
            i_g = ig.mean(dim=-1)
            ax.bar(torch.arange(len(tokens)), i_g[0])
        else:
            i_g = ig  # DH: default is None
            ax.bar(torch.arange(len(tokens)), i_g)
        
        y0, y1 = ax.get_ylim()
        #ax.axvline(x=focus, ymin=y0, ymax=y1, c="r", alpha=0.5)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_ylabel("IG")
        plt.tight_layout()
        if plot:
            plt.pause(0.01)
        return fig, ax

    @staticmethod
    def prediction_histograms(hist, start_words, samples, horizon, plot=False):
        max_text_len = 50
        sampledict = {"fontsize": 12, }#"fontweight": "bold"}
        fig, ax = plt.subplots(len(hist), 1, sharex=True, figsize=(7, 10))
        for i, (word, pred_dist, sample) in enumerate(zip(start_words, hist, samples)):
            sample_text = " ".join(list(sample))
            if len(sample_text) > max_text_len:
                sample_text = sample_text[: max_text_len - 3] + "..."
            h_vals = ax[i].hist(
                pred_dist, range=(0, horizon), bins=horizon // 2, alpha=0.3
            )
            ax[i].set_ylabel(
                word,
                rotation=0,
                #fontweight="bold",
                fontsize=12,
                labelpad=2,
                horizontalalignment="right",
            )
            ax[i].set_yticks([])
            ax[i].set_xlim([0, horizon])
            ax[i].set_xticks([0, horizon // 2, horizon])
            sample_text_y = ax[i].get_ylim()[-1] / 2
            sample_text_y -= sample_text_y * 0.2
            ax[i].text(x=2, y=sample_text_y, s=sample_text, fontdict=sampledict)
        plt.subplots_adjust(left=0.2)
        if plot:
            plt.pause(0.01)
        return fig, ax


if __name__ == "__main__":
    model, dm, args = load()

    evaluation_model = TurnGPTEval(model, dm.tokenizer)

    if torch.cuda.is_available():
        evaluation_model = evaluation_model.to("cuda")

    chkpt_root = split(args.tokenizer)[0]
    savepath = join(chkpt_root, "evaluation")
    makedirs(savepath, exist_ok=True)

    test_dataloader = get_dataloader(dm, args)
    # DH:
    print(">> Model, data ok; savepath:", savepath)
    #input(">> [1/1] Press any key to continue")

    if args.perplexity:
        # ce_loss, ppl = perplexity(model, dm, args)
        ce_loss, ppl = evaluation_model.cross_entropy(test_dataloader)
        print("split: ", args.split)
        print("avg CE loss: ", ce_loss)
        print("ppl (nats): ", ppl)
        write_txt([f"ce_loss: {ce_loss}", f"ppl: {ppl}"], join(savepath, "loss.txt"))

    if args.classification:
        score = evaluation_model.classification(test_dataloader)
        fig, ax = Plots.bacc(score, plot=args.plot)
        fig.savefig(join(savepath, f"bacc_{args.datasets}_{args.split}.png"))
        torch.save(score, join(savepath, f"bacc_{args.datasets}_{args.split}.pt"))
        pgm = score["positive_guesses"].mean()
        pgs = score["positive_guesses"].std()
        ngm = score["negative_guesses"].mean()
        ngs = score["negative_guesses"].std()
        print("Pos: ", pgm, pgs)
        print("Pos: ", ngm, ngs)

    n_context = 4

    if args.context_ablation:
        predictions = evaluation_model.context_ablation(
            test_dataloader, n_context=n_context, max_batch_size=20
        )
        context_score = []
        for i in range(n_context + 1):
            context_score.append(
                fscores(predictions["positive"][i], predictions["negative"][i])
            )
        predictions["context_score"] = context_score
        prob_thresh = 0.03
        thresh_score = []
        for context, score in enumerate(context_score):
            i = torch.where(score["cutoffs"] == prob_thresh)
            thresh_score.append(score["bacc"][i])

        fig, ax = Plots.context_ablation(thresh_score, plot=args.plot)
        fig.savefig(join(savepath, f"abl_{args.datasets}_{args.split}.png"))
        torch.save(predictions, join(savepath, f"abl_{args.datasets}_{args.split}.pt"))

    prob_thresh = 0.2
    if args.context_attention:
        context_attention_score = evaluation_model.context_attention(
            test_dataloader, prob_thresh, n_context
        )
        fig, ax = Plots.context_attention(context_attention_score, plot=args.plot)
        fig.savefig(join(savepath, f"att_{args.datasets}_{args.split}.png"))
        torch.save(
            context_attention_score,
            join(savepath, f"att_{args.datasets}_{args.split}.pt"),
        )

    if args.prediction_hist:
        n_samples = 1000
        horizon = 50
        batch_size = 40
        # a space at the beginning of a sentence: https://huggingface.co/transformers/v3.3.1/model_doc/gpt2.html#transformers.GPT2Tokenizer
        # turns = [
        #     " yesterday we met in the park",
        #     " okay when will you meet again",
        #     " tomorrow",
        #     "",
        # ]
        turns_list = [
            [
                " can i help you",
                " i want that brown dog to go away",
                "",
            ],
            [
                " can i help you",
                " i want to have that brown dog go away",
                "",
            ],
            [
                " can i help you",
                " i want to let that brown dog go away",
                "",
            ],
        ]
        for i, turns in enumerate(turns_list):
            input_ids, speaker_ids = turns_to_turngpt_tensors(
                turns, dm.tokenizer, explicit_turn_shift=True
            )
            prediction = evaluation_model.prediction_histogram(
                input_ids,
                speaker_ids,
                dm.tokenizer,
                n_samples=n_samples,
                batch_size=batch_size,
                horizon=horizon,
                start_after_first_turn=False,
            )
            print("-"*20)
            for j, sw in enumerate(prediction["start_words"]):
                print(sw, prediction["samples"][j])
            
            fig, ax = Plots.prediction_histograms(
                prediction["prediction_distribution"],
                prediction["start_words"],
                prediction["samples"],
                horizon=horizon,
                plot=args.plot,
            )
            fig.savefig(join(savepath, f"pred_hist{i}.png"))


    # DH NOTE: IG calculation is only available for pretrained model (not for mini).
    if args.context_ig:
        context_ig = evaluation_model.context_IG(
            test_dataloader, prob_thresh, n_context, m=70
        )
        fig, ax = Plots.context_attention(
            context_ig, ylim=[-0.5, 2], ylabel="IG", plot=args.plot
        )
        fig.savefig(join(savepath, f"ig_{args.datasets}_{args.split}.png"))
        torch.save(
            context_ig,
            join(savepath, f"ig_{args.datasets}_{args.split}.pt"),
        )
    
    # added by DH
    if args.trp_sample:
        turns_list = [
            [
                " yesterday we met in the park",
                " okay when will you meet again",
                " tomorrow",
                "",
            ], 
            [
                " can i help you",
                " i want that brown dog to go away",
                "",
            ],
            [
                " can i help you",
                " i want to have that brown dog go away",
                "",
            ],
            [
                " can i help you",
                " i want to let that brown dog go away",
                "",
            ],
        ]
        highlightx_list = ["tomorrow", "dog", "dog", "dog"]
        for i, turns in enumerate(turns_list):
            input_ids, speaker_ids = turns_to_turngpt_tensors(
                turns, dm.tokenizer, explicit_turn_shift=True
            )
            trp = evaluation_model.get_trp(input_ids, speaker_ids)
            #print(">> trp:", trp)
            print(">> input_ids:", input_ids.squeeze(0))
            full_string = input_ids_to_token(input_ids, dm.tokenizer)
            input_tokens = [token for token in full_string.split() if token!=""]
            # NOTE: trp: tensor([[], [], ...]) here we want the first in the batch only
            fig, ax = Plots.trp_sample(
                trp.cpu().detach().numpy()[0], input_tokens, highlightx=highlightx_list[i]
            )
            fig.savefig(join(savepath, f"trp_sample_{i}.png"))
    
    # added by DH
    if args.word_ig:
        turns_list = [
            [
                " yesterday we met in the park",
                " okay when will you meet again",
                " tomorrow",
                "",
            ], 
            [
                " can i help you",
                " i want that brown dog to go away",
                "",
            ], 
            [
                " can i help you",
                " i want to have that brown dog go away",
                "",
            ], 
            [
                " can i help you",
                " i want to let that brown dog go away",
                "",
            ],
        ]
        focus_list = [" again", " dog", " dog", " dog"]
        # prepare data
        data_list = []
        for i, turns in enumerate(turns_list):
            input_ids, speaker_ids = turns_to_turngpt_tensors(
                turns, dm.tokenizer, explicit_turn_shift=True
            )
            # specific: focus token id
            focus_id = dm.tokenizer.encode(focus_list[i])
            print(">>", input_ids, focus_id)
            data_list.append( [input_ids, speaker_ids, focus_id] )
        # compute ig
        word_ig, word_ids = evaluation_model.focus_word_IG(
            data_list, m=120 #70
        )
        # represent result
        for i, ig in enumerate(word_ig):
            # res: e.g. tensor([  0.0000, -19.0994, -16.5760, -19.1928,  15.5170])
            # word_ids[i]: e.g. tensor([50257,  7415,   356,  1138,   287])
            print(">>", ig)
            tokens = [dm.tokenizer.decode(tok_id.item()) for tok_id in word_ids[i]]
            print(">>", tokens)#, word_ids[i])
            fig, ax = Plots.integrated_gradient(
                ig, tokens, focus_list[i], 
            )
            fig.savefig(join(savepath, f"custom_word_ig_{i}.png"))
        torch.save(
            word_ig,
            join(savepath, f"custom_word_ig.pt"),
        )

    # added by DH
    if args.false_word_ig:
        # calculate and save
        word_ig, word_ids = evaluation_model.word_IG(
            test_dataloader, prob_thresh, m=120, actual_end=False, 
            save_step=50, tokenizer=dm.tokenizer, savepath=savepath, 
            restore_from=18000, end_step=20000
        )
        # job1: 13600 going to 20000; job2: 20000 going to final

    # added by DH
    if args.true_word_ig:
        # calculate and save
        word_ig, word_ids = evaluation_model.word_IG(
            test_dataloader, prob_thresh, m=120, actual_end=True, 
            save_step=50, tokenizer=dm.tokenizer, savepath=savepath, 
            restore_from=13700, end_step=None
        )

    # added by DH
    if args.data_stat:
        # token_list = evaluation_model.data_statistic(test_dataloader, dm.tokenizer)
        # for item in token_list[:20]:
        #     print(item)
        samples = evaluation_model.sample_data(test_dataloader, dm.tokenizer)
        # [[], [], ..., []]
        with open(join(savepath, f"sample_{args.datasets[0]}.txt" ), mode='w') as f:
            for sample in samples:
                f.write("".join(sample))

    #ans = input("end?")
