
def print_progress_bar(i , max_index, bar_length):
    '''
    Prints a progress bar to the console.

    :param i: current iteration
    :param max_index: total number of iterations
    :param bar_length: length of the progress bar in characters
    '''
    percent = float(i) / max_index
    arrow = '=' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    print(f'Progress: [{arrow}{spaces}] {int(round(percent * 100))}%', end='\r')

    if i == max_index:
        print()  # Move to the next line on completion
