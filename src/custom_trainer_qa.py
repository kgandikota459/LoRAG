import torch
from extra.trainer_qa import QuestionAnsweringTrainer

class CustomQATrainer(QuestionAnsweringTrainer):
    """
    Subclasses the QuestionAnsweringTrainer to manually handle loss calculation
    if labels are provided in a specific format (like 'labels' in generative QA).
    """
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        # 1. Pop the general 'labels' key if it exists (standard for seq2seq/generative models)
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
            
        # 2. Perform the forward pass with remaining inputs
        outputs = model(**inputs)

        # 3. If standard QA labels (start/end positions) are available, calculate loss
        start_positions = inputs.get("start_positions", None)
        end_positions = inputs.get("end_positions", None)
        
        if start_positions is not None and end_positions is not None:
            # Standard extractive QA loss calculation
            start_logits = outputs.get("start_logits")
            end_logits = outputs.get("end_logits")
            loss_fct = torch.nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        elif labels is not None:
             # Handle the 'labels' case if you are doing generative QA (like T5/BART)
             # The HF models for seq2seq QA usually return loss automatically if 'labels' is present,
             # but we check here just in case.
             loss = outputs.get("loss")
             if loss is None:
                 raise ValueError("Model inputs had 'labels' but did not return a 'loss'.")

        else:
            # This is the scenario causing your error during *training*
            # You must have labels present during the training_step
            if self.model.training:
                 raise ValueError(
                     "Cannot train without labels. Inputs received: " + str(inputs.keys()) + 
                     ". Expected 'start_positions' and 'end_positions' (for extractive QA) or 'labels' (for seq2seq QA)."
                 )
            # During evaluation/prediction, it's okay not to have a loss
            loss = None 

        return (loss, outputs) if return_outputs else loss

