
def print_progress_bar(i , max_index, bar_length):
    '''
    Fortschrittsanzeige in der Konsole.

    :param i: jetziger schritt
    :param max_index: gesamtanzahl der schritte
    :param bar_length: länge der anzeige
    '''
    percent = float(i) / max_index
    arrow = '=' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    print(f'Progress: [{arrow}{spaces}] {int(round(percent * 100))}%', end='\r')

    if i == max_index:  #wenn fertig, neue zeile
        print() 
