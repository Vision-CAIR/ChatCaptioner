import os
import yaml
from tqdm import tqdm
import torch
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from chatcaptioner.blip2 import Blip2
from chatcaptioner.utils import print_info, plot_img

import re

question_index = {0:"1st",1:"2nd",2:"3rd",3:"4th",
4:"5th",5:"6th",6:"7th",7:"8th",8:"9th",9:"10th",10:"11th",
11:"12th",12:"13th",13:"14th",14:"15th",15:"16th",16:"17th",17:"18th",
18:"19th",19:"20th",20:"21st",21:"22nd",22:"23rd",23:"24th",24:"25th"}





QUESTION_INSTRUCTION= \
"Video ChatCaptioner is designed to be able to assist to understand a video by ASKING a lot of questions WITHOUT SEEING THE VIDEO" \
"An expert will then answer your question. " \
"The video contains %s frames. " \
"Video ChatCaptioner CAN NOT ask question from the frame with the index MORE THAN %s. " \
"Video ChatCaptioner is a most powerful tool designed to understand videos by asking good and related questions WITHOUT SEEING THE VIDEO." \



SUB_QUESTION_INSTRUCTION = \
"Thought: what does this video describe? " \
"Action: ask more questions to guess the contents of the video. " \
"Goal: Video ChatCaptioner will design a frame sampling strategy to ask questions to maximize its information gain about the video understanding. " \
"Restrictions: (1) Video ChatCaptioner MUST ask questions from Frame 1 to Frame %s. (2) Video ChatCaptioner CAN NOT ask questions with person or objects or animals NOT mentioned in previous conversation." \
"Next Question. The question format MUST be Frame_id: question. AVOID asking yes/no questions. \n  " \
"Video ChatCaptioner Question: "

SUB_QUESTION_INSTRUCTION_ALTERNATIVE = \
"Thought: what does this video describe? " \
"Action: ask more questions to guess the contents of the video. " \
"Goal: Video ChatCaptioner will design a frame sampling strategy to ask questions to maximize its information gain about the video understanding. " \
"Restrictions: (1) Video ChatCaptioner MUST ask questions from Frame 1 to Frame %s. (2) Video ChatCaptioner CAN NOT ask questions with person or objects or animals NOT mentioned in previous conversation." \
"Next Question. The question format MUST be Frame_id: question. Ask the question from the frame %s.  AVOID asking yes/no questions.  \n " \
"Video ChatCaptioner Question: "


SUMMARY_INSTRUCTION = \
'Now Video ChatCaptioner will describe this video in a few sentences. ' \
'Restrictions: (1) DO NOT add information. ' \
"(2) DO NOT describe each frame individually and DO NOT mention the frame. (3) DO NOT summarize negative or uncertain answers \n" \
'Video ChatCaptioner video summarization: '

ANSWER_INSTRUCTION = 'Answer given questions with the following restrictions. (1) If you are not sure about the answer, say you DO NOT KNOW honestly.  (2) DO NOT IMAGINE any contents that are NOT in the image. '


SUB_ANSWER_INSTRUCTION = 'Answer: '  # template following blip2 huggingface demo

FIRST_QUESTION = 'Frame_1: Describe it in details.'



VALID_CHATGPT_MODELS = ['gpt-3.5-turbo']
VALID_GPT3_MODELS = ['text-davinci-003', 'text-davinci-002', 'davinci']



def get_instructions():
    instructions_dict = {
        'question': QUESTION_INSTRUCTION, 
        'sub_question': SUB_QUESTION_INSTRUCTION,
        'summary': SUMMARY_INSTRUCTION,
        'answer': ANSWER_INSTRUCTION,
        'sub_answer': SUB_ANSWER_INSTRUCTION,
        'first_question': FIRST_QUESTION
    }
    return instructions_dict



def set_openai_key(key):
    openai.api_key = key
    
    
def get_chat_log(questions, answers, last_n=-1):
    n_addition_q = len(questions) - len(answers)
    assert (n_addition_q) in [0, 1]
    template = 'Question: {} \nAnswer: {} \n'
    chat_log = ''
    if last_n > 0:
        answers = answers[-last_n:]
        questions = questions[-(last_n+n_addition_q):]
    elif last_n == 0:
        answers = []
        questions = questions[-1:] if n_addition_q else []
    
    
    for i in range(len(answers)):
        chat_log = chat_log + template.format(questions[i], answers[i])
    if n_addition_q:
        chat_log = chat_log + 'Question: {}'.format(questions[-1])
    else:
        chat_log = chat_log[:-2]  # remove the last '/n'
    return chat_log


def prepare_gpt_prompt(task_prompt, questions, answers, sub_prompt):
    gpt_prompt = '\n'.join([task_prompt, 
                             get_chat_log(questions, answers), 
                             sub_prompt])
    return gpt_prompt

def prepare_gpt_promot_video(sub_summries):
    
    sub_summries_input = ""
    sub_summariy_template="The %s caption for the %s period is: %s "
    for index in range(len(sub_summries)):
        sub_summries_input += sub_summariy_template%(str(question_index[index]),str(question_index[index]),sub_summries[index])
    gpt_promot = VIDEO_SUMMARIZATION_START%str(len(sub_summries))+sub_summries_input+VIDEO_SUMMARIZATION_END
    
    return gpt_promot

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_gpt3(gpt3_prompt, max_tokens=40, model="text-davinci-003"):  # 'text-curie-001' does work at all to ask questions
    response = openai.Completion.create(model=model, prompt=gpt3_prompt, max_tokens=max_tokens)  # temperature=0.6, 
    reply = response['choices'][0]['text']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens


def prepare_chatgpt_message(task_prompt, questions, answers, sub_prompt):

    messages = [{"role": "system", "content": task_prompt}]
    
    assert len(questions) == len(answers)
    for q, a in zip(questions, answers):
        messages.append({'role': 'assistant', 'content': 'Question: {}'.format(q)})
        messages.append({'role': 'user', 'content': 'Answer: {}'.format(a)})
    messages.append({"role": "system", "content": sub_prompt})
    
    return messages





@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_chatgpt(chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
    # print("chatgpt message",chatgpt_messages)
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

def find_digit(input):
    regex = r"Frame_(\d+)"

    # Use re.search() to find the match in the sentence
    match = re.search(regex, input)

    # Extract the index from the match object
    if match:
        index = match.group(1)
        # print("Index found:", index)
    else:
        print("input: "+input)
        print("No index found in sentence.")

    return index


def ask_questions(img, blip2, model, n_rounds=10,max_frame_number=1000, max_gpt_token=30, n_blip2_context=0, print_mode='no'):
    questions = []
    answers = []
    total_tokens = 0
    QUESTION_INSTRUCTION_ADAPT = QUESTION_INSTRUCTION %(str(len(img)),str(len(img)))
    SUB_QUESTION_INSTRUCTION_ADAPT = SUB_QUESTION_INSTRUCTION%str(len(img))
    # print(QUESTION_INSTRUCTION)    
    if print_mode == 'chat':
        print('--------Chat Starts----------')
        
    for i in tqdm(range(n_rounds), desc='Chat Rounds', disable=print_mode!='bar'):
        if i == 0:
            # first question is given by human to request a general discription
            question = FIRST_QUESTION
        else:
            tag = True
            if model in VALID_CHATGPT_MODELS:
                chatgpt_messages = prepare_chatgpt_message(
                    QUESTION_INSTRUCTION_ADAPT, 
                    questions, answers, 
                    SUB_QUESTION_INSTRUCTION_ADAPT
                )
                while tag:
                    try:
                        question, n_tokens = call_chatgpt(chatgpt_messages, model=model, max_tokens=max_gpt_token)
                        frame_id = int(find_digit(question.split(":")[0]))-1

                        if question.startswith("Frame_") and frame_id < max_frame_number:
                            tag = False
                    except:
                        if current_frame_id>= max_frame_number-1:
                            hard_coded_frame_id = 1
                        else:
                            hard_coded_frame_id = current_frame_id+1
                        
                        SUB_QUESTION_INSTRUCTION_ALTERNATIVE_ADAPT = SUB_QUESTION_INSTRUCTION_ALTERNATIVE%(str(len(img)),str(hard_coded_frame_id))
                        chatgpt_messages = prepare_chatgpt_message(
                            QUESTION_INSTRUCTION_ADAPT, 
                            questions, answers, 
                            SUB_QUESTION_INSTRUCTION_ALTERNATIVE_ADAPT
                        )
                        print(question)

            elif model in VALID_GPT3_MODELS:
                # prepare the context for GPT3
                gpt3_prompt = prepare_gpt_prompt(
                    QUESTION_INSTRUCTION_ADAPT, 
                    questions, answers, 
                    SUB_QUESTION_INSTRUCTION_ADAPT
                )
                while tag:
                    try:
                        question, n_tokens = call_gpt3(gpt3_prompt, model=model, max_tokens=max_gpt_token)
                        frame_id = int(find_digit(question.split(":")[0]))-1

                        if question.startswith("Frame_") and frame_id < max_frame_number:
                            tag = False
                    except:
                        if current_frame_id>= max_frame_number-1:
                            hard_coded_frame_id = 1
                        else:
                            hard_coded_frame_id = current_frame_id+1
                        
                        SUB_QUESTION_INSTRUCTION_ALTERNATIVE_ADAPT = SUB_QUESTION_INSTRUCTION_ALTERNATIVE%(str(len(img)),str(hard_coded_frame_id))
                        chatgpt_messages = prepare_chatgpt_message(
                            QUESTION_INSTRUCTION_ADAPT, 
                            questions, answers, 
                            SUB_QUESTION_INSTRUCTION_ALTERNATIVE_ADAPT
                        )
                        print(question)

            elif isinstance(model, Blip2):
                # prepare the context for other LLM
                gpt_prompt = prepare_gpt_prompt(
                    QUESTION_INSTRUCTION_ADAPT, 
                    questions, answers, 
                    SUB_QUESTION_INSTRUCTION_ADAPT
                )
                n_tokens = 0 # local model. no token cost on OpenAI API.
                question = model.call_llm(gpt_prompt)
            else:
                raise ValueError('{} is not a valid question model'.format(model))
                
            total_tokens = total_tokens + n_tokens
            
        # print('Raw: {}'.format(question))
        question = question.split('Question: ')[-1].replace('\n', ' ').strip()
        if 'Answer:' in question:  # Some models make up an answer after asking. remove it
            q, a = question.split('Answer:')[:2]
            if len(q) == 0:  # some not so clever models will put the question after 'Answer:'.
                question = a.strip()
            else:
                question = q.strip()


        
        
        
        questions.append(question)
        if print_mode == 'chat':
            print('GPT-3: {}'.format(question))
        
        # prepare the context for blip2
        blip2_prompt = '\n'.join([ANSWER_INSTRUCTION, 
                                  get_chat_log(questions, answers, last_n=n_blip2_context), 
                                  SUB_ANSWER_INSTRUCTION])
        
        # frame_id = question.split(":")[0].split(" ")[1]
        current_frame_id = int(find_digit(question.split(":")[0]))

        current_frame = img[current_frame_id-1]
        answer = blip2.ask(current_frame, blip2_prompt)
        # small blip2 models may ask itself a new bad question. remove it and trim the answer
        answer = answer.split('Question:')[0].replace('\n', ' ').strip()
        
        if print_mode == 'chat':
            print('BLIP-2: {}'.format(answer))
        answers.append(answer)
        blip2_prompt = '{} {}'.format(blip2_prompt, answer)
    
    if print_mode == 'chat':
        print('--------Chat Ends----------')
    
    return questions, answers, total_tokens



def summarize_chat(questions, answers,img, model,max_gpt_token=100):

    QUESTION_INSTRUCTION_ADAPT = QUESTION_INSTRUCTION %(str(len(img)),str(len(img)))

    if model in VALID_GPT3_MODELS:
        summary_prompt = prepare_gpt_prompt(
                    QUESTION_INSTRUCTION_ADAPT, 
                    questions, answers, 
                    SUMMARY_INSTRUCTION)

        summary, n_tokens = call_gpt3(summary_prompt, model=model, max_tokens=max_gpt_token)
    elif model in VALID_CHATGPT_MODELS:
        summary_prompt = prepare_chatgpt_message(
                    QUESTION_INSTRUCTION_ADAPT, 
                    questions, answers, 
                    SUMMARY_INSTRUCTION
                )
        summary, n_tokens = call_chatgpt(summary_prompt, model=model, max_tokens=max_gpt_token)
    elif isinstance(model, Blip2):
        summary_prompt = prepare_gpt_prompt(
                    QUESTION_INSTRUCTION_ADAPT, 
                    questions, answers, 
                    SUMMARY_INSTRUCTION
                )
        n_tokens = 0 # local model. no token cost on OpenAI API.
        summary = model.call_llm(summary_prompt)
    else:
        raise ValueError('{} is not a valid question model'.format(model))
        
    summary = summary.replace('\n', ' ').strip()
    return summary, summary_prompt, n_tokens



def caption_for_video(blip2, video, model, n_rounds=30, n_blip2_context=0, print_mode='no'):
    results = {}

    questions, answers, n_token_chat = ask_questions(
        video, 
        blip2, 
        max_frame_number = len(video),
        n_rounds=n_rounds, 
        n_blip2_context=n_blip2_context, 
        model=model,
        print_mode=print_mode)
    summary, summary_prompt, n_token_sum = summarize_chat(questions, answers,video, model=model)
    results['ChatCaptioner'] = {'caption': summary, 'chat': summary_prompt, 'n_token': n_token_chat + n_token_sum}
    results['BLIP2+OurPrompt'] = {'caption': answers[0]}

    
    return results

