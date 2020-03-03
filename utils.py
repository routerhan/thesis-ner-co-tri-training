import os
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm, trange


from preprocessor.preprocessor import *
from transformers import BertTokenizer, BertForTokenClassification


def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters

def flat_accuracy(preds, labels):
    return np.sum(np.array(preds)==np.array(labels))/len(labels)

def annot_confusion_matrix(valid_tags, pred_tags):

    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    # Create header from unique tags
    header = sorted(list(set(valid_tags + pred_tags)))

    # Calculate the actual confusion matrix
    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    # Final formatting touches for the string output
    mat_formatted = [header[i] + "\t" + str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content


class BertTrainer:
    def __init__(self, bert_model, max_len, batch_size, full_finetuning, sentences, labels, tag2idx, idx2tag):
        # Specify device data for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Devices available: {}".format(self.device))
        self.FULL_FINETUNING = full_finetuning
        self.BATCH_SIZE = batch_size
        self.idx2tag = idx2tag

        # Initialize PRETRAINED TOKENIZER
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
        # Initialize PRETRAINED MODEL
        self.model = BertForTokenClassification.from_pretrained(bert_model, num_labels=len(tag2idx))
        self.model.to(self.device)

        # Tokenize the sentences
        self.tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
        # Get input id of token
        self.input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in self.tokenized_texts],
                                        maxlen=max_len, dtype="long", truncating="post", padding="post")
        # Get tags of labels with id
        self.tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                                        maxlen=max_len, value=tag2idx["O"], padding="post",
                                        dtype="long", truncating="post")

        # Get attention mask from BERT label
        self.attention_masks = [[float(i>0) for i in ii] for ii in self.input_ids]

    def get_train_and_vaild_dataloader(self):
        # Split the dataset for 10% validation
        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(self.input_ids, self.tags, 
                                                                random_state=42, test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(self.attention_masks, self.input_ids,
                                                                random_state=42, test_size=0.1)

        # Convert dataset to torch tensors
        tr_inputs = torch.tensor(tr_inputs)
        val_inputs = torch.tensor(val_inputs)
        tr_tags = torch.tensor(tr_tags)
        val_tags = torch.tensor(val_tags)
        tr_masks = torch.tensor(tr_masks)
        val_masks = torch.tensor(val_masks)

        # Load and prepare data, define dataloaders
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.BATCH_SIZE)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.BATCH_SIZE)
        print("Loaded training and validation data into DataLoaders.")
        return train_dataloader, valid_dataloader

    def train_and_save_model(self, epochs=4, max_grad_norm=1.0, learning_rate=3e-5):
        model = self.model
        # Set hyperparameters (optimizer, weight decay, learning rate)
        optimizer_grouped_parameters = get_hyperparameters(model, self.FULL_FINETUNING)
        optimizer = Adam(optimizer_grouped_parameters, lr=learning_rate)
        print("Initialized optimizer and set hyperparameters.")

        # Get data loader
        train_dataloader, valid_dataloader = self.get_train_and_vaild_dataloader()

        # Start fine-tuning model, set epochs and max_grad_norm
        epoch = 0
        for _ in trange(epochs, desc="Epoch"):
            epoch += 1

            print("Starting training loop.")
            model.train()
            tr_loss, tr_accuracy = 0, 0
            nb_tr_examples, nb_tr_steps = 0, 0
            tr_preds, tr_labels = [], []

            for step, batch in enumerate(train_dataloader):
                # add batch to gpus
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # Forward pass
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss, tr_logits = outputs[:2]

                # Backward pass
                loss.backward()

                # Compute train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=max_grad_norm
                )

                # Update parameters
                optimizer.step()
                model.zero_grad()

            # print train loss per epoch
            print("Train loss: {}".format(tr_loss/nb_tr_steps))


            # Validation loop
            print("Starting validation loop.")

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions, true_labels = [], []

            for batch in valid_dataloader:

                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels,
                    )
                    tmp_eval_loss, logits = outputs[:2]

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.append(label_ids)

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1

                
            # Evakuate loss, acc, conf. matrix and report on dev set.
            pred_tags = [self.idx2tag[p_i] for p in predictions for p_i in p]
            valid_tags = [self.idx2tag[l_li] for l in true_labels[0] for l_li in l]
            cl_report = classification_report(valid_tags, pred_tags)
            conf_mat = annot_confusion_matrix(valid_tags, pred_tags)
            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_steps

            # Report metrics
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy))
            print("Classification Report:\n {}".format(cl_report))
            print("Confusion Matrix:\n {}".format(conf_mat))
            print("F1-Score: {}".format(flat_accuracy(pred_tags, valid_tags)))