from typing import List

def handle_command_flag(text, flag):
    '''
    Handle command flag in text.
    Command flags always start with "$$". 
    if next character is "-" followed by an integer, remove the last n lines
    if next character is "+" followed by an integer, remove the first n lines
    '''
    if flag[2] == "-":
        n = int(flag[3:])
        lines = text.split("\n")
        text = "\n".join(lines[:-n])
    elif flag[2] == "+":
        n = int(flag[3:])
        lines = text.split("\n")
        text = "\n".join(lines[n:])
    else:
        raise Exception("Invalid command flag.")

    return text
    
def cutoff_leading_text(text:str, flag_list:List[str]):
    for flag in flag_list:
        if flag[:2] == "$$":
            text = handle_command_flag(text, flag)
            return text
        search_result = text.find(flag)
        if search_result != -1:
            return text[search_result:]
        
    raise Exception("No cutoff leading text flag found in text.")

def cutoff_trailing_text(text:str, flag_list:List[str]):
    for flag in flag_list:
        if flag[:2] == "$$":
            text = handle_command_flag(text, flag)
            return text
        search_result = text.rfind(flag)
        if search_result != -1:
            return text[:search_result]
        
    raise Exception("No cutoff trailing text flag found in text.")