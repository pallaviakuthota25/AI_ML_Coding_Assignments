"""
This program is build with Flan-T5-XL LLM to be able to answer the final question using the output from the previous questions as in-context learning/few-shot learning. 

Consider three related questions from a search session: Question 1, Question 2, Question 3
1. Answer to Question 1 needs to be generated. 
2. Answer to Question 2 needs to be generated with the answer to Question 1 as one-shot example / context. 
3. Answer to Question 3 needs to be generated with the answer to Question 2 as one-shot example / context.
4. Answer to Question 3 will be either YES or NO and nothing else.


> The program accepts three parameters provided as a command line input. 
> The three inputs represent the questions.
> The output of the first two question is Generation based whereas the last question output is deterministic i.e. its either YES or NO.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.


Syntax: python template.py <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in America?"
Terminal Output: NO

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in India?"
Terminal Output: YES

You are expected to create some examples of your own to test the correctness of your approach.

ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

"""
* * * Changes allowed from here  * * * 
"""

def llm_function(model,tokenizer,questions):
    '''
    The steps are given for your reference:

    1. Generate answer for the first question.
    2. Generate answer for the second question use the answer for first question as context.
    3. Generate a deterministic output either 'YES' or 'NO' for the third question using the context from second question.  
    5. Clean output and return.
    6. Output is case-sensative: YES or NO
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    question1, question2, question3 = questions

    # Step 1: Generate answer for the first question with proper instruction prompting
    prompt1 = f"Answer the following question with detailed information:\n\nQuestion: {question1}\nAnswer:"
    inputs1 = tokenizer(prompt1, return_tensors="pt")
    outputs1 = model.generate(**inputs1, max_length=100, do_sample=False)
    answer1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
    
    # Step 2: Few-shot prompting for second question using first Q&A as example
    few_shot_prompt2 = f"""Answer questions using the context provided. Use the previous question and answer as reference to answer the current question.

    Example:
    Question: {question1}
    Answer: {answer1}

    Now answer this related question:
    Question: {question2}
    Answer:"""
    
    inputs2 = tokenizer(few_shot_prompt2, return_tensors="pt")
    outputs2 = model.generate(**inputs2, max_length=150, do_sample=False)
    answer2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
    
    # Step 3: Few-shot prompting for YES/NO question with clear instructions and examples
    few_shot_prompt3 = f"""Answer questions with only YES or NO based on the context provided. Look at the example below:

    Example:
    Question: Where was Rabindranath Tagore born?
    Answer: Rabindranath Tagore was born in Calcutta, Bengal Presidency, British India.
    Question: Is it in India?
    Answer: YES

    Another example:
    Question: Where was Rabindranath Tagore born?
    Answer: Rabindranath Tagore was born in Calcutta, Bengal Presidency, British India.
    Question: Is it in America?
    Answer: NO

    Now use this context to answer with only YES or NO:
    Question: {question2}
    Answer: {answer2}
    Question: {question3}
    Answer:"""
    
    inputs3 = tokenizer(few_shot_prompt3, return_tensors="pt")
    outputs3 = model.generate(**inputs3, max_length=50, do_sample=False)
    answer3 = tokenizer.decode(outputs3[0], skip_special_tokens=True)
    
    # Step 4: Clean output and ensure it's YES or NO
    answer3_cleaned = answer3.strip().upper()
    
    # Use regex to find YES or NO in the response
    yes_no_match = re.search(r'\b(YES|NO)\b', answer3_cleaned)
    if yes_no_match:
        final_output = yes_no_match.group(1)
    else:
        # If no clear YES/NO found, analyze the response content
        if any(word in answer3_cleaned.lower() for word in ['yes', 'true', 'correct', 'right', 'indeed', 'exactly']):
            final_output = "YES"
        else:
            final_output = "NO"
    
    return final_output

"""
ALERT: * * * No changes are allowed below this comment  * * *
"""

if __name__ == '__main__':

    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]
    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,questions)
    print(out.strip())

    """ End to call """