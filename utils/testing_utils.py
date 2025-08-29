import os
import re
import tempfile
import subprocess
from typing import Dict, Any, List, Optional


def detect_python_version(code: str) -> str:
    """
    Detect whether the Python code is Python 2 or Python 3.
    
    Args:
        code (str): The Python source code to analyze
        
    Returns:
        str: 'python2', 'python3', or 'python' (if uncertain)
    """
    code_lower = code.lower().strip()
    
    # Strong indicators for Python 2
    python2_patterns = [
        r'print\s+[^(]',  # print statement without parentheses
        r'raw_input\s*\(',  # raw_input function
        r'xrange\s*\(',  # xrange function
        r'\.iteritems\s*\(',  # dict.iteritems()
        r'\.iterkeys\s*\(',  # dict.iterkeys()
        r'\.itervalues\s*\(',  # dict.itervalues()
        r'unicode\s*\(',  # unicode function
        r'basestring',  # basestring type
        r'long\s*\(',  # long type
        r'exec\s+[^(]',  # exec statement
        r'raise\s+\w+,',  # raise exception, message syntax
        r'<>\s*',  # <> operator
        r'from\s+__future__\s+import',  # future imports (usually Python 2)
    ]
    
    # Strong indicators for Python 3
    python3_patterns = [
        r'print\s*\(',  # print function with parentheses
        r'input\s*\(',  # input function (was raw_input in Python 2)
        r'range\s*\(',  # range function (was xrange in Python 2)
        r'\.items\s*\(',  # dict.items() (without iter prefix)
        r'\.keys\s*\(',  # dict.keys() (without iter prefix)
        r'\.values\s*\(',  # dict.values() (without iter prefix)
        r'nonlocal\s+',  # nonlocal keyword
        r'async\s+def',  # async function
        r'await\s+',  # await keyword
        r'yield\s+from',  # yield from syntax
        r'@\w+\.setter',  # property setter (more common in Python 3)
        r'typing\.',  # typing module imports
        r'f["\']',  # f-string literals
        r'pathlib\.',  # pathlib module
    ]
    
    # Count matches
    python2_score = sum(1 for pattern in python2_patterns if re.search(pattern, code, re.IGNORECASE))
    python3_score = sum(1 for pattern in python3_patterns if re.search(pattern, code, re.IGNORECASE))
    
    # Additional heuristics
    lines = code.split('\n')
    
    # Check for print statements vs print functions
    print_statements = len(re.findall(r'print\s+[^(]', code))
    print_functions = len(re.findall(r'print\s*\(', code))
    
    if print_statements > print_functions:
        python2_score += 2
    elif print_functions > print_statements:
        python3_score += 2
    
    # Check for division behavior comments or explicit imports
    if 'from __future__ import division' in code:
        python2_score += 1
    
    # Check for string literals
    if re.search(r'[ub]["\']', code, re.IGNORECASE):  # b"" or u"" strings
        python2_score += 1
    
    # Check for type annotations (Python 3.5+)
    if re.search(r':\s*\w+\s*=', code) or re.search(r'def\s+\w+\([^)]*:\s*\w+', code):
        python3_score += 2
    
    # Determine version
    if python2_score > python3_score:
        return 'python2'
    elif python3_score > python2_score:
        return 'python3'
    else:
        # If scores are equal or both are 0, return generic 'python'
        return 'python'


def detect_programming_language(code: str) -> str:
    """
    Detect the programming language of the given code.
    
    Args:
        code (str): The source code to analyze
        
    Returns:
        str: The detected language ('python2', 'python3', 'python', 'java', 'c', 'cpp', or 'unknown')
    """
    code_lower = code.lower().strip()
    
    # Java detection patterns
    java_patterns = [
        r'public\s+class\s+\w+',
        r'public\s+static\s+void\s+main',
        r'System\.out\.print',
        r'Scanner\s+\w+\s*=\s*new\s+Scanner',
        r'import\s+java\.',
        r'package\s+\w+',
    ]
    
    # C++ detection patterns
    cpp_patterns = [
        r'#include\s*<iostream>',
        r'#include\s*<vector>',
        r'#include\s*<string>',
        r'#include\s*<algorithm>',
        r'#include\s*<queue>',
        r'#include\s*<stack>',
        r'#include\s*<map>',
        r'#include\s*<set>',
        r'#include\s*<unordered_map>',
        r'#include\s*<unordered_set>',
        r'std::',
        r'using\s+namespace\s+std',
        r'cout\s*<<',
        r'cin\s*>>',
        r'endl',
        r'vector\s*<',
        r'string\s+\w+',
        r'class\s+\w+\s*{',
        r'public\s*:',
        r'private\s*:',
        r'protected\s*:',
    ]
    
    # C detection patterns  
    c_patterns = [
        r'#include\s*<stdio\.h>',
        r'#include\s*<stdlib\.h>',
        r'#include\s*<string\.h>',
        r'#include\s*<math\.h>',
        r'int\s+main\s*\(',
        r'printf\s*\(',
        r'scanf\s*\(',
        r'#define\s+\w+',
        r'malloc\s*\(',
        r'free\s*\(',
    ]
    
    # Python detection patterns
    python_patterns = [
        r'def\s+\w+\s*\(',
        r'class\s+\w+\s*\(?.*\)?:',
        r'import\s+\w+',
        r'from\s+\w+\s+import',
        r'if\s+__name__\s*==\s*["\']__main__["\']',
        r'print\s*\(',
        r'input\s*\(',
    ]
    
    # Count matches for each language
    java_score = sum(1 for pattern in java_patterns if re.search(pattern, code, re.IGNORECASE))
    cpp_score = sum(1 for pattern in cpp_patterns if re.search(pattern, code, re.IGNORECASE))
    c_score = sum(1 for pattern in c_patterns if re.search(pattern, code, re.IGNORECASE))
    python_score = sum(1 for pattern in python_patterns if re.search(pattern, code, re.IGNORECASE))
    
    # Strong indicators for C++
    if re.search(r'#include\s*<iostream>', code, re.IGNORECASE) or \
       re.search(r'std::', code) or \
       re.search(r'using\s+namespace\s+std', code, re.IGNORECASE):
        cpp_score += 5
    
    # Additional heuristics
    # Check for typical Python indentation patterns
    lines = code.split('\n')
    indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
    if indented_lines > len(lines) * 0.2:  # If more than 20% of lines are indented
        python_score += 2
    
    # Check for semicolons (common in Java/C/C++, rare in Python)
    semicolon_lines = sum(1 for line in lines if line.strip().endswith(';'))
    if semicolon_lines > len(lines) * 0.3:
        java_score += 1
        c_score += 1
        cpp_score += 1
    
    # Check for curly braces (Java/C/C++)
    if '{' in code and '}' in code:
        java_score += 1
        c_score += 1
        cpp_score += 1
    
    # Determine the language with highest score
    scores = {'java': java_score, 'cpp': cpp_score, 'c': c_score, 'python': python_score}
    max_score = max(scores.values())
    
    if max_score == 0:
        return 'python'
    
    # Return the language with the highest score
    for lang, score in scores.items():
        if score == max_score:
            # If it's Python, detect the specific version
            if lang == 'python':
                return detect_python_version(code)
            return lang
    
    return 'python'


def run_testcase_stdio(
    solution_code: str,
    test_case: dict,
    timeout: int = 10
) -> dict:
    """
    Run a stdio-based test case against a solution and return the result.
    Supports Python 2, Python 3, Java, C, and C++ code with automatic language detection.
    
    Args:
        solution_code (str): The solution code to test (should use input()/print())
        test_case (dict): {"input": ..., "output": ...}
        timeout (int): Maximum execution time in seconds
    Returns:
        dict: {"passed": bool, "stdout": str, "stderr": str, "error": str, "language": str}
    """
    import tempfile
    import subprocess
    import os
    import shutil

    # Detect the programming language
    language = detect_programming_language(solution_code)
    
    if language == 'unknown':
        return {
            "passed": False,
            "stdout": '',
            "stderr": '',
            "error": "Could not detect programming language",
            "language": "unknown"
        }

    # Create a temporary directory for compilation if needed
    temp_dir = tempfile.mkdtemp()
    
    try:
        if language in ['python', 'python2', 'python3']:
            return _run_python_stdio(solution_code, test_case, timeout, temp_dir, language)
        elif language == 'java':
            return _run_java_stdio(solution_code, test_case, timeout, temp_dir)
        elif language == 'c':
            return _run_c_stdio(solution_code, test_case, timeout, temp_dir)
        elif language == 'cpp':
            return _run_cpp_stdio(solution_code, test_case, timeout, temp_dir)
        else:
            return {
                "passed": False,
                "stdout": '',
                "stderr": '',
                "error": f"Unsupported language: {language}",
                "language": language
            }
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def _run_python_stdio(solution_code: str, test_case: dict, timeout: int, temp_dir: str, language: str) -> dict:
    """Run Python code with stdio testing."""
    temp_file_path = os.path.join(temp_dir, 'solution.py')
    
    with open(temp_file_path, 'w') as f:
        # Add appropriate imports based on Python version
        if language == 'python2':
            # Python 2 specific imports and compatibility
            f.write(solution_code)
        
        elif language == 'python3' or language == 'python':
            # Common imports for all Python versions
            f.write('import random\n')
            f.write('import functools\n')
            f.write('import itertools\n')
            f.write('import collections\n')
            f.write('import heapq\n')
            f.write('import bisect\n')
            f.write('import operator\n')
            f.write('import re\n')
            f.write('import string\n')
            f.write('import sys\n')
            f.write('import time\n')
            f.write('import math\n')
            f.write('import datetime\n')
            f.write('from typing import *\n')
            f.write('from functools import *\n')
            f.write('from collections import *\n')
            f.write('from itertools import *\n')
            f.write('from heapq import *\n')
            f.write('from string import *\n')
            f.write('from operator import *\n')
            f.write('from math import *\n')
            f.write('from bisect import *\n')
            f.write('from re import *\n\n')
            f.write(solution_code)

    # Determine which Python interpreter to use
    if language == 'python2':
        python_cmd = 'python2'
    elif language == 'python3':
        python_cmd = 'python3'
    else:
        # Default to python if version is uncertain
        python_cmd = 'python3'

    try:
        #print(python_cmd, temp_file_path)
        result = subprocess.run(
            [python_cmd, temp_file_path],
            input=test_case.get('input', ''),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            cwd=temp_dir
        )
        
        actual_output = result.stdout.strip()
        expected_output = str(test_case.get('output', '')).strip()
        # requirements for passed test case
        passed = (actual_output == expected_output) and (result.stderr == '') and (result.returncode == 0) and (actual_output != '')
        
        return {
            "passed": passed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": '' if passed else 'Output mismatch',
            "language": language
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": '',
            "stderr": f"Timeout after {timeout} seconds",
            "error": "Test execution timed out",
            "language": language
        }
    except FileNotFoundError:
        # If the specific Python version is not available, try fallback
        fallback_cmd = 'python' if python_cmd != 'python' else 'python3'
        try:
            result = subprocess.run(
                [fallback_cmd, temp_file_path],
                input=test_case.get('input', ''),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
                cwd=temp_dir
            )
            
            actual_output = result.stdout.strip()
            expected_output = str(test_case.get('output', '')).strip()
            passed = (actual_output == expected_output)
            
            return {
                "passed": passed,
                "stdout": result.stdout,
                "stderr": result.stderr + f"\nNote: Used {fallback_cmd} instead of {python_cmd}",
                "error": '' if passed else 'Output mismatch',
                "language": language
            }
        except Exception as e:
            return {
                "passed": False,
                "stdout": '',
                "stderr": f"Python interpreter not found: {python_cmd} or {fallback_cmd}",
                "error": f"Python interpreter not found: {str(e)}",
                "language": language
            }
    except Exception as e:
        return {
            "passed": False,
            "stdout": '',
            "stderr": str(e),
            "error": str(e),
            "language": language
        }


def _run_java_stdio(solution_code: str, test_case: dict, timeout: int, temp_dir: str) -> dict:
    """Run Java code with stdio testing."""
    # Extract class name from the code
    class_match = re.search(r'public\s+class\s+(\w+)', solution_code)
    if not class_match:
        return {
            "passed": False,
            "stdout": '',
            "stderr": '',
            "error": "Could not find public class declaration in Java code",
            "language": "java"
        }
    
    class_name = class_match.group(1)
    java_file_path = os.path.join(temp_dir, f'{class_name}.java')
    
    # Add common imports if not already present
    common_imports = [
        'import java.util.*;',
        'import java.io.*;', 
        'import java.math.*;',
        'import java.text.*;',
        'import java.util.regex.*;'
    ]
    
    # Check which imports are already in the code
    existing_imports = set()
    for line in solution_code.split('\n'):
        line = line.strip()
        if line.startswith('import '):
            existing_imports.add(line)
    
    # Add missing common imports
    imports_to_add = []
    for imp in common_imports:
        if imp not in existing_imports:
            # Check if more specific import already exists
            package = imp.split()[1].replace('.*;', '')
            specific_exists = any(existing.startswith(f'import {package}.') for existing in existing_imports)
            if not specific_exists:
                imports_to_add.append(imp)
    
    with open(java_file_path, 'w') as f:
        # Write imports first
        for imp in imports_to_add:
            f.write(imp + '\n')
        if imports_to_add:
            f.write('\n')
        f.write(solution_code)
    
    try:
        # Compile Java code
        compile_result = subprocess.run(
            ['javac', java_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            cwd=temp_dir
        )
        
        if compile_result.returncode != 0:
            return {
                "passed": False,
                "stdout": '',
                "stderr": compile_result.stderr,
                "error": f"Compilation failed: {compile_result.stderr}",
                "language": "java"
            }
        
        # Run Java code
        result = subprocess.run(
            ['java', class_name],
            input=test_case.get('input', ''),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            cwd=temp_dir
        )
        
        actual_output = result.stdout.strip()
        expected_output = str(test_case.get('output', '')).strip()
        passed = (actual_output == expected_output) and (result.stderr == '')
        
        return {
            "passed": passed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": '' if passed else 'Output mismatch',
            "language": "java"
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": '',
            "stderr": f"Timeout after {timeout} seconds",
            "error": "Test execution timed out",
            "language": "java"
        }
    except Exception as e:
        return {
            "passed": False,
            "stdout": '',
            "stderr": str(e),
            "error": str(e),
            "language": "java"
        }


def _run_c_stdio(solution_code: str, test_case: dict, timeout: int, temp_dir: str) -> dict:
    """Run C code with stdio testing."""
    c_file_path = os.path.join(temp_dir, 'solution.c')
    executable_path = os.path.join(temp_dir, 'solution')
    
    # Add common includes if not already present
    common_includes = [
        '#include <stdio.h>',
        '#include <stdlib.h>',
        '#include <string.h>',
        '#include <math.h>',
        '#include <ctype.h>',
        '#include <limits.h>',
        '#include <stdbool.h>',
        '#include <assert.h>'
    ]
    
    # Check which includes are already in the code
    existing_includes = set()
    for line in solution_code.split('\n'):
        line = line.strip()
        if line.startswith('#include'):
            existing_includes.add(line)
    
    # Add missing common includes
    includes_to_add = []
    for inc in common_includes:
        if inc not in existing_includes:
            includes_to_add.append(inc)
    
    with open(c_file_path, 'w') as f:
        # Write includes first
        for inc in includes_to_add:
            f.write(inc + '\n')
        if includes_to_add:
            f.write('\n')
        f.write(solution_code)
    
    try:
        # Compile C code with math library
        compile_result = subprocess.run(
            ['gcc', '-o', executable_path, c_file_path, '-lm'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            cwd=temp_dir
        )
        
        if compile_result.returncode != 0:
            return {
                "passed": False,
                "stdout": '',
                "stderr": compile_result.stderr,
                "error": f"Compilation failed: {compile_result.stderr}",
                "language": "c"
            }
        
        # Run C executable
        result = subprocess.run(
            [executable_path],
            input=test_case.get('input', ''),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            cwd=temp_dir
        )
        
        actual_output = result.stdout.strip()
        expected_output = str(test_case.get('output', '')).strip()
        passed = (actual_output == expected_output) and (result.stderr == '')
        
        return {
            "passed": passed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": '' if passed else 'Output mismatch',
            "language": "c"
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": '',
            "stderr": f"Timeout after {timeout} seconds",
            "error": "Test execution timed out",
            "language": "c"
        }
    except Exception as e:
        return {
            "passed": False,
            "stdout": '',
            "stderr": str(e),
            "error": str(e),
            "language": "c"
        }


def _run_cpp_stdio(solution_code: str, test_case: dict, timeout: int, temp_dir: str) -> dict:
    """Run C++ code with stdio testing."""
    cpp_file_path = os.path.join(temp_dir, 'solution.cpp')
    executable_path = os.path.join(temp_dir, 'solution')
    
    # Add common includes if not already present
    common_includes = [
        '#include <iostream>',
        '#include <vector>',
        '#include <string>',
        '#include <algorithm>',
        '#include <queue>',
        '#include <stack>',
        '#include <map>',
        '#include <set>',
        '#include <unordered_map>',
        '#include <unordered_set>',
        '#include <cmath>',
        '#include <climits>',
        '#include <cstring>',
        '#include <cstdlib>',
        '#include <cctype>',
        '#include <cassert>'
    ]
    
    # Check which includes are already in the code
    existing_includes = set()
    for line in solution_code.split('\n'):
        line = line.strip()
        if line.startswith('#include'):
            existing_includes.add(line)
    
    # Add missing common includes
    includes_to_add = []
    for inc in common_includes:
        if inc not in existing_includes:
            includes_to_add.append(inc)
    
    # Check if using namespace std is present
    has_using_namespace = 'using namespace std' in solution_code
    
    with open(cpp_file_path, 'w') as f:
        # Write includes first
        for inc in includes_to_add:
            f.write(inc + '\n')
        if includes_to_add:
            f.write('\n')
        
        # Add using namespace std if not present
        if not has_using_namespace:
            f.write('using namespace std;\n\n')
        
        f.write(solution_code)
    
    try:
        # Compile C++ code
        compile_result = subprocess.run(
            ['g++', '-o', executable_path, cpp_file_path, '-std=c++17'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            cwd=temp_dir
        )
        
        if compile_result.returncode != 0:
            return {
                "passed": False,
                "stdout": '',
                "stderr": compile_result.stderr,
                "error": f"Compilation failed: {compile_result.stderr}",
                "language": "cpp"
            }
        
        # Run C++ executable
        result = subprocess.run(
            [executable_path],
            input=test_case.get('input', ''),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
            cwd=temp_dir
        )
        
        actual_output = result.stdout.strip()
        expected_output = str(test_case.get('output', '')).strip()
        passed = (actual_output == expected_output)
        
        return {
            "passed": passed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": '' if passed else 'Output mismatch',
            "language": "cpp"
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": '',
            "stderr": f"Timeout after {timeout} seconds",
            "error": "Test execution timed out",
            "language": "cpp"
        }
    except Exception as e:
        return {
            "passed": False,
            "stdout": '',
            "stderr": str(e),
            "error": str(e),
            "language": "cpp"
        }



# misc functions
def run_testcase(
    solution_code: str, 
    test_case: str, 
    entry_point: Optional[str] = None, 
    timeout: int = 10
    ) -> Dict[str, Any]:
    """
    Run a test case against a solution and return the result.
    
    Args:
        solution_code (str): The solution code to test
        test_case (str): The test case (assertion statement)
        entry_point (str): The entry point for testing (e.g., "Solution().shortestDistanceAfterQueries")
        timeout (int): Maximum execution time in seconds
        
    Returns:
        Dict[str, Any]: Dictionary containing test results
    """
    # Create a temporary file to run the test
    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+', delete=False) as temp_file:
        # Write solution code and test case to the file
        temp_file.write('import random\n')
        temp_file.write('import functools\n')
        temp_file.write('import itertools\n')
        temp_file.write('import collections\n')
        temp_file.write('import heapq\n')
        temp_file.write('import bisect\n')
        temp_file.write('import operator\n')
        temp_file.write('import re\n')
        temp_file.write('import string\n')
        temp_file.write('import sys\n')
        temp_file.write('import time\n')
        temp_file.write('import math\n')
        temp_file.write('import datetime\n')
        
        temp_file.write('from typing import *\n')
        temp_file.write('from functools import *\n')
        temp_file.write('from collections import *\n')
        temp_file.write('from itertools import *\n')
        temp_file.write('from heapq import *\n')
        temp_file.write('from string import *\n')
        temp_file.write('from operator import *\n')
        temp_file.write('from math import *\n')
        temp_file.write('from bisect import *\n')
        temp_file.write('from operator import *\n')
        temp_file.write('from re import *\n')
        
        temp_file.write(solution_code + '\n\n')
        
        
        # Setup the candidate function using the entry point
        if entry_point:
            temp_file.write(f'# Using provided entry point\n')
            temp_file.write(f'candidate = {entry_point}\n\n')
        else:
            temp_file.write(f'# No entry point provided, trying to infer\n')
            temp_file.write(f'try:\n')
            temp_file.write(f'    # Try to find Solution class\n')
            temp_file.write(f'    if "Solution" in globals():\n')
            temp_file.write(f'        solution_instance = Solution()\n')
            temp_file.write(f'        # Try to find a method in the Solution class\n')
            temp_file.write(f'        for attr_name in dir(solution_instance):\n')
            temp_file.write(f'            if attr_name.startswith("__"):\n')
            temp_file.write(f'                continue\n')
            temp_file.write(f'            if callable(getattr(solution_instance, attr_name)):\n')
            temp_file.write(f'                candidate = getattr(solution_instance, attr_name)\n')
            temp_file.write(f'                break\n')
            temp_file.write(f'except Exception as e:\n')
            temp_file.write(f'    print(f"Setup error: {{e}}")\n')
            temp_file.write(f'    candidate = None\n\n')
        
        # Run the test case
        temp_file.write('try:\n')
        temp_file.write(f'    {test_case}\n')
        temp_file.write('    print("PASS")\n')
        temp_file.write('except Exception as e:\n')
        temp_file.write('    print(f"FAIL: {e}")\n')
        
        temp_file_path = temp_file.name
        
    try:
        # Run the test file
        result = subprocess.run(
            ['python', temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        
        passed = "PASS" in result.stdout
        error_msg = ""
        
        if not passed:
            # Extract error message if any
            if "FAIL:" in result.stdout:
                error_msg = result.stdout.split("FAIL:", 1)[1].strip()
            else:
                error_msg = result.stdout.strip() or result.stderr.strip()
        
        return {
            "passed": passed,
            "error": error_msg,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error": "Test execution timed out",
            "stdout": "",
            "stderr": f"Timeout after {timeout} seconds"
        }
    
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "stdout": "",
            "stderr": str(e)
        }
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        


def extract_coverage_percentage(coverage_output: str) -> float:
    """
    Extract coverage percentage from coverage report output.
    """
    # Look for pattern like "TOTAL    123    45    63%"
    lines = coverage_output.strip().split('\n')
    for line in lines:
        if 'TOTAL' in line or 'solution.py' in line:
            # Extract percentage (last column ending with %)
            parts = line.split()
            for part in reversed(parts):
                if part.endswith('%'):
                    try:
                        return float(part[:-1])
                    except ValueError:
                        continue
    return 0.0


def evaluate_test_coverage(
    solution_code: str, 
    test_cases: List[str], 
    entry_point: Optional[str] = None, 
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Run test coverage for a solution and return the result.
    """
    import tempfile
    import shutil
    import re
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Write solution.py
        solution_path = os.path.join(temp_dir, 'solution.py')
        with open(solution_path, 'w') as f:
            # Add necessary imports
            f.write('import random\n')
            f.write('import functools\n')
            f.write('import itertools\n')
            f.write('import collections\n')
            f.write('import heapq\n')
            f.write('import bisect\n')
            f.write('import operator\n')
            f.write('import re\n')
            f.write('import string\n')
            f.write('import sys\n')
            f.write('import time\n')
            f.write('import math\n')
            f.write('import datetime\n')
            f.write('from typing import *\n')
            f.write('from functools import *\n')
            f.write('from collections import *\n')
            f.write('from itertools import *\n')
            f.write('from heapq import *\n')
            f.write('from string import *\n')
            f.write('from operator import *\n')
            f.write('from math import *\n')
            f.write('from bisect import *\n')
            f.write('from re import *\n\n')
            
            # Write the solution code
            f.write(solution_code + '\n')
        
        # Write test.py
        test_path = os.path.join(temp_dir, 'test.py')
        with open(test_path, 'w') as f:
            f.write('from solution import *\n\n')
            
            # Setup the candidate function
            if entry_point:
                f.write(f'candidate = {entry_point}\n\n')
            else:
                f.write('# Try to find Solution class\n')
                f.write('if "Solution" in globals():\n')
                f.write('    solution_instance = Solution()\n')
                f.write('    # Find the first non-private method\n')
                f.write('    for attr_name in dir(solution_instance):\n')
                f.write('        if not attr_name.startswith("__") and callable(getattr(solution_instance, attr_name)):\n')
                f.write('            candidate = getattr(solution_instance, attr_name)\n')
                f.write('            break\n\n')
            
            # Write test function
            f.write('def test_solution():\n')
            if len(test_cases) > 0:
                for test_case in test_cases:
                    f.write(f'    {test_case}\n')
            else:
                f.write('    pass\n')
            
            f.write('\nif __name__ == "__main__":\n')
            f.write('    test_solution()\n')
        
        # Run coverage
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Run coverage
            coverage_run = subprocess.run(
                ['python', '-m', 'coverage', 'run', 'test.py'],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Get coverage report
            coverage_report = subprocess.run(
                ['python', '-m', 'coverage', 'report', '-m'],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Extract coverage percentage
            coverage_percentage = extract_coverage_percentage(coverage_report.stdout)
            
            return {
                "coverage_percentage": coverage_percentage,
                "coverage_output": coverage_report.stdout,
                "run_output": coverage_run.stdout,
                "run_error": coverage_run.stderr,
                "report_error": coverage_report.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "coverage_percentage": 0.0,
                "error": "Coverage measurement timed out",
                "coverage_output": "",
                "run_output": "",
                "run_error": f"Timeout after {timeout} seconds",
                "report_error": ""
            }
        except Exception as e:
            return {
                "coverage_percentage": 0.0,
                "error": str(e),
                "coverage_output": "",
                "run_output": "",
                "run_error": str(e),
                "report_error": ""
            }
        finally:
            os.chdir(original_dir)
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def extract_mutation_score(mutation_log: str) -> float:
    # Look for pattern like "surviving mutants: 13 (17.33%)"
    pattern = r'surviving mutants: \d+ \((\d+\.?\d*)%\)'
    match = re.search(pattern, mutation_log)
    if match:
        survive_rate = float(match.group(1))
    else:
        survive_rate = 0.0
    
    return (100.0 - survive_rate) / 100.0


def evaluate_test_mutation_score(
    solution_code: str, 
    test_cases: List[str], 
    entry_point: Optional[str] = None, 
    timeout: int = 120,
) -> Dict[str, Any]:
    """
    Run test coverage for a solution and return the result.
    """
    if len(test_cases) == 0:
        return {
            "error": "No valid test cases",
            "mutation_score": 0.0,
            "mutation_log": ""
        }
    
    import tempfile
    import shutil
    import re
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()  # Save current directory at the beginning
    
    try:
        # Write solution.py
        solution_path = os.path.join(temp_dir, 'solution.py')
        with open(solution_path, 'w') as f:
            # Add necessary imports
            f.write('import random\n')
            f.write('import functools\n')
            f.write('import itertools\n')
            f.write('import collections\n')
            f.write('import heapq\n')
            f.write('import bisect\n')
            f.write('import operator\n')
            f.write('import re\n')
            f.write('import string\n')
            f.write('import sys\n')
            f.write('import time\n')
            f.write('import math\n')
            f.write('import datetime\n')
            f.write('from typing import *\n')
            f.write('from functools import *\n')
            f.write('from collections import *\n')
            f.write('from itertools import *\n')
            f.write('from heapq import *\n')
            f.write('from string import *\n')
            f.write('from operator import *\n')
            f.write('from math import *\n')
            f.write('from bisect import *\n')
            f.write('from re import *\n\n')
            
            # Write the solution code
            f.write(solution_code + '\n')
        
        # Write test.py
        test_path = os.path.join(temp_dir, 'test_solution.py')
        with open(test_path, 'w') as f:
            f.write('from solution import *\n\n')
            
            # Setup the candidate function
            if entry_point:
                f.write(f'candidate = {entry_point}\n\n')
            else:
                f.write('# Try to find Solution class\n')
                f.write('if "Solution" in globals():\n')
                f.write('    solution_instance = Solution()\n')
                f.write('    # Find the first non-private method\n')
                f.write('    for attr_name in dir(solution_instance):\n')
                f.write('        if not attr_name.startswith("__") and callable(getattr(solution_instance, attr_name)):\n')
                f.write('            candidate = getattr(solution_instance, attr_name)\n')
                f.write('            break\n\n')
            
            # Write test function
            f.write('def test_solution():\n')
            if len(test_cases) > 0:
                for test_case in test_cases:
                    f.write(f'    {test_case}\n')
            else:
                f.write('    pass\n')
            
            f.write('\nif __name__ == "__main__":\n')
            f.write('    test_solution()\n')
        
        # Write config.toml
        config_path = os.path.join(temp_dir, 'config.toml')
        with open(config_path, 'w') as f:
            f.write('[cosmic-ray]\n')
            f.write('module-path = "solution.py"\n')
            f.write('timeout = 10.0\n')
            f.write('excluded-modules = []\n')
            f.write('test-command = "pytest test_solution.py"\n')
            f.write('\n[cosmic-ray.distributor]\n')
            f.write('name = "local"\n')
        
        # Run mutation testing
        # Change to temp directory
        os.chdir(temp_dir)
        
        try:
            # initialize cosmic-ray session
            subprocess.run(['cosmic-ray', 'init', 'config.toml', 'result.sqlite'],
                           capture_output=True,
                           text=True,
                           timeout=timeout)
            # baselining
            baseline_result = subprocess.run(
                ['cosmic-ray', 'baseline', 'config.toml'],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            # if baselining fails -> return 0.0
            if baseline_result.returncode != 0:
                return {
                    "error": "Baseline failed",
                    "mutation_score": 0.0,
                    "mutation_log": baseline_result.stderr
                }
            else:
                try:
                    exec_result =subprocess.run(
                        ['cosmic-ray', 'exec', 'config.toml', 'result.sqlite'],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                except subprocess.TimeoutExpired as e:
                    # if timeout -> evaluate mutation score based on the tried mutations
                    pass
                # get mutation score
                mutation_result = subprocess.run(
                    ['cr-report', 'result.sqlite', '--show-pending'],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                mutation_log = mutation_result.stdout
                mutation_score = extract_mutation_score(mutation_log)
                
                return {
                    "mutation_log": mutation_log,
                    "mutation_score": mutation_score,
                    "error": ""
                }
        
        except subprocess.TimeoutExpired as e:
            return {
                "error": str(e),
                "mutation_score": 0.0,
                "mutation_log": ""
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "mutation_score": 0.0,
                "mutation_log": ""
            }
    
    finally:
        # Restore original directory
        try:
            os.chdir(original_dir)
        except:
            pass  # Ignore errors when restoring directory
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass  # Ignore errors when cleaning up temp directory
