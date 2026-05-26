import os

TRAING_CASE_1 = {'id': 'physical-commonsense-1353',
'category': 'Usage Purpose',
'image_id': 'commonsense-physical-commonsense-148',
'question': 'What is the purpose of the item made of wood in the picture?',
'choices': [
"To display information",
"To use as a writing surface",
"To sit on it",
"To hold the computer screen"
],
'context': '',
'answer': 'B',
'rationale': '''We can see in the picture that there is a monitor, which indicates the presence of a computer.
Similarly, we can see a keyboard, which shows that this is a computer workstation.
The presence of wood suggests that it is being used as a desk or writing surface.
From the given clues and inferences, we can conclude that the purpose of the item made of wood is to use as a writing surface with the help of the keyboard.
Hence, option B is correct.
Option A is incorrect because the monitor is the device used to display information, not the wood.
Option C is also incorrect because the wood is not shaped to be sat upon.
Option D is incorrect because the stand attached to the back of the monitor is holding the screen, not the wood.
Therefore, option B is the correct answer.''',
'split': 'train',
'image': 'data\\images\\physical-commonsense-1353.png',
'domain': 'commonsense',
'topic': 'physical-commonsense'}

demo_image_path = os.path.join('./data/m3cot/data/images', TRAING_CASE_1['id'] + '.png')

example_user_text = '''Below is one solved example.
Learn the reasoning style from the example, but answer the final question only based on the final image and its options.

Example question:
Question: {}
Options:
A. {}
B. {}
C. {}
D. {}
'''.format(TRAING_CASE_1['question'], *TRAING_CASE_1['choices'])


zero_shot_prompt_template = '''Question: {}
Options:
'''

mcot_induct_0 = '''Question: {}
Options:
A. {}
B. {}
C. {}
D. {}
'''.format(TRAING_CASE_1['question'], *TRAING_CASE_1['choices'])


mcot_induct_1 = '''Let's think step by step. First, this region shows a monitor and a keyboard, which indicates that this is a computer workstation.'''
mcot_induct_2 = '''Second, the wooden object appears to be the desk surface in front of the workstation, so it is being used as a writing or working surface.'''
mcot_induct_3 = '''Therefore, option B is correct. Option A is incorrect because the monitor displays information, not the wood. Option C is incorrect because the wood is not shaped for sitting. Option D is incorrect because the monitor stand, not the wood itself, holds the screen.'''
mcot_induct_4 = '''Answer: {}\n'''.format(TRAING_CASE_1['answer'])


generation_config = {
    'do_sample': False,
    'temperature': 0.8,
    'top_p': 0.9,
    'max_new_tokens': 512,
    'repetition_penalty': 1.1
}
