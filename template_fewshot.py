import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

def llm_function(model, tokenizer, questions):
    """
    1. Answer Q1.
    2. Answer Q2 using (Q1,A1) as a one-shot example.
    3. Answer Q3 with YES or NO only, using (Q2,A2) as context, greedy decoding.
    """

    q1, q2, q3 = questions

    # Step 1: answer question 1
    prompt1 = f"Q: {q1}\nA:"
    inputs = tokenizer(prompt1, return_tensors="pt")
    out1 = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )
    a1 = tokenizer.decode(out1[0], skip_special_tokens=True).strip()

    # Step 2: answer question 2, with (Q1,A1) as context
    prompt2 = (
        f"Q: {q1}\nA: {a1}\n"
        f"Q: {q2}\nA:"
    )
    inputs = tokenizer(prompt2, return_tensors="pt")
    out2 = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )
    a2 = tokenizer.decode(out2[0], skip_special_tokens=True).strip()

    # Step 3: answer question 3, forcing YES or NO only
    prompt3 = (
        f"Q: {q2}\nA: {a2}\n"
        f"Q: {q3}\nAnswer with YES or NO only:"
    )
    inputs = tokenizer(prompt3, return_tensors="pt")
    out3 = model.generate(
        **inputs,
        max_new_tokens=2,
        do_sample=False,
        num_beams=1
    )
    a3_raw = tokenizer.decode(out3[0], skip_special_tokens=True).strip()

    # Normalize into exactly “YES” or “NO”
    low = a3_raw.lower()
    if low.startswith("yes"):
        return "YES"
    elif low.startswith("no"):
        return "NO"
    else:
        # fallback if model hallucinated
        return "YES" if "yes" in low else "NO"

if __name__ == '__main__':

    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]
    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    torch.manual_seed(42)
    out = llm_function(model, tokenizer, questions)
    print(out.strip())

    # --- Self-tests (uncomment to run locally) ---
    # tests = [
    #     # basic geography
    #     (
    #         "What is the capital of France?",
    #         "Which country is Paris the capital of?",
    #         "Is that country in Europe?",
    #         "YES"
    #     ),
    #     # literature
    #     (
    #         "Who wrote The Great Gatsby?",
    #         "Where was that author born?",
    #         "Was it in Canada?",
    #         "NO"
    #     ),
    #     # science
    #     (
    #         "What is H2O commonly called?",
    #         "Is this substance a liquid at room temperature?",
    #         "Does it boil at below 50°C?",
    #         "NO"
    #     ),
    #     # history
    #     (
    #         "Who was the first President of the United States?",
    #         "Where was he born?",
    #         "Was it in the state of Virginia?",
    #         "YES"
    #     ),
    # ]
    #
    # for q1, q2, q3, expected in tests:
    #     res = llm_function(model, tokenizer, [q1, q2, q3])
    #     print(f"Q1: {q1}")
    #     print(f"Q2: {q2}")
    #     print(f"Q3: {q3} -> {res} (expected {expected})")
    #     assert res == expected
    # print("All self-tests passed!")