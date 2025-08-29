import re
import ast
import sys
from typing import List

# parsing function to extract python codeblock from the response
def extract_python_code(response: str) -> str:
    return response.split('```python')[-1].split('```')[0].strip()


# parsing function to extract test cases from the response (UTRL format)
def extract_test_cases_stdio(response: str):
    pattern = re.compile(
        r"```[\s\S]*?Input:\n(?P<input>[\s\S]*?)\nOutput:\n(?P<output>[\s\S]*?)```", re.MULTILINE
    )

    test_cases = []
    for match in pattern.finditer(response):
        inp = match.group("input").strip()
        out = match.group("output").strip()
        test_cases.append({"input": inp, "output": out})
    return test_cases



# Utility function for test case extraction in CURE format
def modify(c):
    c = c.replace("plaintext\n", "")
    c = c.replace("\\n", "\n")
    if not c.endswith("\n"):
        c += "\n"
    return c


def parse_test_case_cure(full_output: str):
    # First, try extracting with the updated triple-backtick pattern
    pattern_input_backticks = r'\*\*Test Input:\*\*\s*```(.*?)```'
    pattern_output_backticks = r'\*\*Test Output:\*\*\s*```(.*?)```'
    matches_input = re.findall(pattern_input_backticks, full_output, re.DOTALL)
    matches_output = re.findall(pattern_output_backticks, full_output, re.DOTALL)

    # For Test Input: either use the updated triple-backtick version or fallback to plain text
    if matches_input:
        test_input = [modify(matches_input[-1].lstrip('\n'))]
    else:
        # Fallback pattern without backticks: capture until **Test Output:**
        pattern_input_plain = r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)'
        matches_input_plain = re.findall(pattern_input_plain, full_output, re.DOTALL)
        if matches_input_plain:
            test_input = [modify(matches_input_plain[-1].strip())]
        else:
            test_input = []
    
    # For Test Output: either use the updated triple-backtick version or fallback to plain text
    if matches_output:
        test_output = [modify(matches_output[-1].lstrip('\n'))]
    else:
        # Fallback: capture until the **Explanation:** marker or end-of-string
        pattern_output_plain = r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|$)'
        matches_output_plain = re.findall(pattern_output_plain, full_output, re.DOTALL)
        if matches_output_plain:
            test_output = [modify(matches_output_plain[-1].strip())]
        else:
            test_output = []
    
    # Also extract from the last occurrence of **Test Input:** to the end
    index = full_output.rfind("**Test Input:**")
    if index != -1:
        example_text = [full_output[index:]]
    else:
        example_text = []
    
    # If any essential piece is missing, return empties
    if example_text == [] or test_input == [] or test_output == []:
        return "", ""
    
    return {'input': test_input[0].strip().strip('"'), 'output': test_output[0].strip().strip('"')}


def extract_test_cases_cure(list_of_responses: List[str]):
    test_cases = []
    for response in list_of_responses:
        test_cases.append(parse_test_case_cure(response))
    return test_cases
