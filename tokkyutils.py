def ask_yes_no(question):
    while True:
        answer = input(question + " [Y/N]: ").strip().lower()
        if answer in ('y', 'yes', 'Y'):
            return True
        elif answer in ('n', 'no', 'N'):
            return False
        else:
            print("Please enter Y or N.")