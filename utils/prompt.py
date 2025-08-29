
###### TACO benchmark (stdio format test) ######

# -------------------------- SYSTEM PROMPTS --------------------------
SOLUTION_GENERATION_SYSTEM_PROMPT_STDIO = """
You are an expert Python programmer.
Based on the problem description, solve the given coding problem efficiently.
Think step by step, and write a Python solution that solves the problem. 
Specifically, follow the format below:

<reasoning> 
Write your reasoning here. 
</reasoning>

```python
Write your Python solution here. Your solution should be able to run with the stdio format input.
```
"""

SOLUTION_GENERATION_WITHOUT_THINK_SYSTEM_PROMPT_STDIO = """
You are an expert Python programmer.
Based on the problem description, solve the given coding problem efficiently.
Write a Python solution that solves the problem. 
Specifically, follow the format below:

```python
Write your Python solution here. Your solution should be able to run with the stdio format input.
```
"""



TEST_GENERATION_SYSTEM_PROMPT_STDIO = """
You are an expert Python programmer capable at generating test cases for Python programming tasks.
Given a programming task, generate several test cases and corresponding reasoning.
Each test case should be independent of each other, and sharply cover the corner cases, so that arbitrary faulty code solutions can be detected.
Before you implement each test case, you must think deeply about the input arguments that verify the extreme edge cases, and reason about the expected output.
After completing the reasoning process between, generate test case in stdio format.
Specifically, your output should follow the format below:


<reasoning>
Reasoning for test case 1
First, reason about the input arguments that can discriminate wrong code solution (e.g., edge cases). 
Note that the input arguments should tests the aspects independent to the previously generated test cases. 
Then, reason about the expected output for the input arguments based on the problem description.
</reasoning>
```
Input:
stdio format input 1

Output:
stdio format output 1
```

<reasoning>
Reasoning for test case 2
First, reason about the input arguments that can discriminate wrong code solution (e.g., edge cases). 
Note that the input arguments should tests the aspects independent to the previously generated test cases. 
Then, reason about the expected output for the input arguments based on the problem description.
</reasoning>
```
Input:
stdio format input 2

Output:
stdio format output 2
```
...

<reasoning>
Reasoning for test case 12
First, reason about the input arguments that can discriminate wrong code solution (e.g., edge cases). 
Note that the input arguments should tests the aspects independent to the previously generated test cases. 
Then, reason about the expected output for the input arguments based on the problem description.
</reasoning>
```
Input:
stdio format input 12

Output:
stdio format output 12
```

Ensure followings:
1. Do not include solution code in your response, and generate 12 test cases in total.
2. For each test case, provide a detailed rationale and reasoning behind the test case.
3. Each test case should be independent of each other, and should not contain duplicate test cases.
"""


TEST_GENERATION_WITHOUT_THINK_SYSTEM_PROMPT_STDIO = """
You are an expert Python programmer capable at generating test cases for Python programming tasks.
Given a programming task, generate several test cases.
Each test case should be independent of each other, and sharply cover the corner cases, so that arbitrary faulty code solutions can be detected.
Specifically, your output should follow the format below:

```
Input:
stdio format input 1

Output:
stdio format output 1
```

```
Input:
stdio format input 2

Output:
stdio format output 2
```
...

```
Input:
stdio format input 12

Output:
stdio format output 12
```

Ensure followings:
1. Do not include solution code in your response, and generate 12 test cases in total.
2. Each test case should be independent of each other, and should not contain duplicate test cases.
"""


# -------------------------- PROMPTS --------------------------
SOLUTION_GENERATION_PROMPT_STDIO = """
Here is the problem description:

{problem_query}

/no_think
""" 

SOLUTION_GENERATION_WITHOUT_THINK_PROMPT_STDIO = """
Here is the problem description:

{problem_query}

/no_think
"""


TEST_GENERATION_PROMPT_STDIO = """
Here is the problem description:

{problem_query}

Based on comprehensive reasoning, generate comprehensive unit test involving several test cases for the given problem.
The test cases should cover various edge cases, corner cases, and normal cases, at the same time, functionally correct.

/no_think
"""

TEST_GENERATION_WITHOUT_THINK_PROMPT_STDIO = """
Here is the problem description:

{problem_query}

generate comprehensive unit test involving several test cases for the given problem.
The test cases should cover various edge cases, corner cases, and normal cases, at the same time, functionally correct.

/no_think
"""



# -------------------------- CURE PROMPTS --------------------------

TEST_GENERATION_PROMPT_CURE = """
Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation). 
This is the problem:

{problem_query}

You need to provide a new test example. 
A good test example should be completely accurate and conform to the problem’s format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code. 
Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. 
For example, start by designing an input you can reliably handle, then compute the output step by step.
If you’re unsure about the output, revise or re-design the input to ensure accuracy.
Finally, after completing these previous thinking and derivation steps, you MUST put your final test example in the following format:
**Test Input:** "input here"
**Test Output:** "output here"
**Explanation:** explanation here.
"""

TEST_GENERATION_SYSTEM_PROMPT_CURE = """
You are a helpful assistant help user generate test examples for coding tasks.
"""