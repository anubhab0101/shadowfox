import tkinter as tk
from tkinter import messagebox
import random

WORD_LISTS = {
    "EASY": [
        "CAT", "DOG", "SUN", "MOON", "FISH", "BOOK", "TREE", "CAR", "BALL", "RUN",
        "JUMP", "SING", "PLAY", "EAT", "READ", "BLUE", "RED", "GREEN", "TALL", "SHORT",
        "HAPPY", "SAD", "HOT", "COLD", "FAST", "SLOW", "CITY", "PARK", "FARM", "OCEAN"
    ],
    "MEDIUM": [
        "PYTHON", "GUITAR", "ORANGE", "TABLET", "FARMER", "DOCTOR", "BIRDIE", "FLOWER", "RAINBOW", "MOUNTAIN",
        "KEYBOARD", "BICYCLE", "PICTURE", "FURNACE", "JOURNAL", "LIBRARY", "WEATHER", "HISTORY", "SCIENCE", "EXPLORE",
        "FOREST", "RIVER", "CASTLE", "MONKEY", "PENCIL", "DESERT", "VOLCANO", "ZEBRA", "SCHOOL", "TRAFFIC"
    ],
    "HARD": [
        "PROGRAMMING", "ALGORITHM", "DIFFICULTY", "ENIGMATIC", "ZEPHYR", "QUASAR", "JUXTAPOSE", "KALEIDOSCOPE",
        "MNEMONIC", "PHOENIX", "RHAPSODY", "SYNCHRONIZE", "XENOPHOBIA", "WHISPERING", "CRYPTOGRAPHY", "PARADOXICAL",
        "ABYSS", "VORTEX", "SYZYGY", "FLUMMOX", "HYPOTHESIS", "JUBILANT", "OBFUSCATE", "QUINTESSENCE", "SERENDIPITY",
        "TRANQUILITY", "UBIQUITOUS", "VICARIOUS", "WINSOME", "YIELDING"
    ]
}

DIFFICULTY_SETTINGS = {
    "EASY": {
        "tries": 10,
        "hints_per_game": 2,
        "letters_per_hint": 1,
        "initial_letters_to_reveal": 1
    },
    "MEDIUM": {
        "tries": 7,
        "hints_per_game": 2,
        "letters_per_hint": 1,
        "initial_letters_to_reveal": (1, 2) 
    },
    "HARD": {
        "tries": 5,
        "hints_per_game": 1,
        "letters_per_hint": 1,
        "initial_letters_to_reveal": (2, 3) 
    }
}

class HangmanGUI:
    """
    A class to create and manage the Hangman game GUI using Tkinter.
    """
    def __init__(self, master):
        self.master = master
        master.title("Hangman Game")
        master.geometry("700x650")
        master.config(bg="#f0f0f0")

        self.secret_word = ""
        self.word_display = []
        self.guessed_letters = set()
        self.guessed_words = set()
        self.current_tries = 0
        self.initial_max_tries = 0
        self.hints_remaining = 0
        self.letters_per_hint = 0
        self.difficulty_var = tk.StringVar(value="EASY")

        self.setup_ui()
        self.start_new_game()

    def setup_ui(self):
        """
        Sets up all the widgets for the game's user interface.
        """
        tk.Label(self.master, text="HANGMAN", font=("Arial", 36, "bold"), bg="#f0f0f0", fg="#4CAF50").pack(pady=10)

        difficulty_frame = tk.Frame(self.master, bg="#f0f0f0")
        difficulty_frame.pack(pady=5)
        tk.Label(difficulty_frame, text="Difficulty:", font=("Arial", 12), bg="#f0f0f0", fg="#333333").pack(side=tk.LEFT)
        for text, mode in [("Easy", "EASY"), ("Medium", "MEDIUM"), ("Hard", "HARD")]:
            rb = tk.Radiobutton(
                difficulty_frame, text=text, variable=self.difficulty_var, value=mode,
                command=self.start_new_game,  # <-- FIX: This command restarts the game on selection.
                font=("Arial", 10), bg="#f0f0f0", fg="#333333", selectcolor="#d4edda", indicatoron=0,
                borderwidth=2, width=8, relief=tk.RAISED
            )
            rb.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.master, width=250, height=250, bg="white", bd=2, relief=tk.SUNKEN)
        self.canvas.pack(pady=10)

        self.word_label_var = tk.StringVar()
        tk.Label(self.master, textvariable=self.word_label_var, font=("Courier", 32, "bold"), bg="#f0f0f0", fg="#0056b3").pack(pady=10)

        self.tries_label_var = tk.StringVar()
        tk.Label(self.master, textvariable=self.tries_label_var, font=("Arial", 14), bg="#f0f0f0", fg="#c0392b").pack()

        self.hints_label_var = tk.StringVar()
        tk.Label(self.master, textvariable=self.hints_label_var, font=("Arial", 14), bg="#f0f0f0", fg="#2980b9").pack()

        self.guessed_letters_var = tk.StringVar()
        tk.Label(self.master, textvariable=self.guessed_letters_var, font=("Arial", 12), bg="#f0f0f0", fg="#555555", wraplength=600).pack()
        
        self.guessed_words_var = tk.StringVar()
        tk.Label(self.master, textvariable=self.guessed_words_var, font=("Arial", 12), bg="#f0f0f0", fg="#555555", wraplength=600).pack()

        input_frame = tk.Frame(self.master, bg="#f0f0f0")
        input_frame.pack(pady=15)

        self.guess_entry = tk.Entry(input_frame, width=15, font=("Arial", 16), justify='center')
        self.guess_entry.pack(side=tk.LEFT, padx=5)
        self.guess_entry.bind("<Return>", self.make_guess_event) # Bind Enter key to guess

        self.guess_button = tk.Button(input_frame, text="Guess", command=self.make_guess, font=("Arial", 14), bg="#2ecc71", fg="white", activebackground="#27ae60", relief=tk.RAISED, bd=2)
        self.guess_button.pack(side=tk.LEFT, padx=5)

        self.hint_button = tk.Button(input_frame, text="Hint", command=self.get_hint, font=("Arial", 14), bg="#3498db", fg="white", activebackground="#2980b9", relief=tk.RAISED, bd=2)
        self.hint_button.pack(side=tk.LEFT, padx=5)

        self.new_game_button = tk.Button(self.master, text="New Game", command=self.start_new_game, font=("Arial", 14), bg="#e67e22", fg="white", activebackground="#d35400", relief=tk.RAISED, bd=2)
        self.new_game_button.pack(pady=10)

    def get_game_parameters(self, difficulty):
        """
        Selects a word and sets game parameters based on the chosen difficulty.
        """
        settings = DIFFICULTY_SETTINGS[difficulty]
        word = random.choice(WORD_LISTS[difficulty])
        word_completion = ["_" for _ in word]
        guessed_letters = set()

        initial_reveal_count = settings["initial_letters_to_reveal"]
        if isinstance(initial_reveal_count, tuple):
            initial_reveal_count = random.randint(initial_reveal_count[0], initial_reveal_count[1])

        unique_word_letters = list(set(word))
        num_initial_reveals = min(initial_reveal_count, len(unique_word_letters))
        
        if num_initial_reveals > 0:
            letters_to_initially_reveal = random.sample(unique_word_letters, num_initial_reveals)
            for char_to_reveal in letters_to_initially_reveal:
                for i, char_in_word in enumerate(word):
                    if char_in_word == char_to_reveal:
                        word_completion[i] = char_in_word
                guessed_letters.add(char_to_reveal)

        return (word.upper(), settings["tries"], settings["hints_per_game"],
                settings["letters_per_hint"], word_completion, guessed_letters)

    def start_new_game(self):
        """
        Resets the game state for a new round.
        """
        chosen_difficulty = self.difficulty_var.get()
        
        self.secret_word, self.initial_max_tries, self.hints_remaining, \
        self.letters_per_hint, self.word_display, self.guessed_letters = \
            self.get_game_parameters(chosen_difficulty)
        
        self.current_tries = self.initial_max_tries
        self.guessed_words = set()

        self.update_display()
        self.draw_hangman_figure()
        self.guess_entry.delete(0, tk.END)
        self.guess_entry.config(state=tk.NORMAL)
        self.guess_button.config(state=tk.NORMAL)
        self.hint_button.config(state=tk.NORMAL if self.hints_remaining > 0 else tk.DISABLED)
        self.guess_entry.focus_set()

    def update_display(self):
        """
        Updates all labels on the screen to reflect the current game state.
        """
        self.word_label_var.set(" ".join(self.word_display))
        self.tries_label_var.set(f"Tries Left: {self.current_tries} / {self.initial_max_tries}")
        self.hints_label_var.set(f"Hints Left: {self.hints_remaining}")
        self.guessed_letters_var.set(f"Guessed Letters: {', '.join(sorted(list(self.guessed_letters)))}")
        self.guessed_words_var.set(f"Guessed Words: {', '.join(sorted(list(self.guessed_words)))}")

    def draw_hangman_figure(self):
        """
        Draws the hangman figure on the canvas based on the number of incorrect guesses.
        """
        self.canvas.delete("all")
        incorrect_guesses_made = self.initial_max_tries - self.current_tries
        
        self.canvas.create_line(50, 240, 150, 240, width=2) # Base
        self.canvas.create_line(100, 240, 100, 40, width=2)  # Pole
        self.canvas.create_line(100, 40, 200, 40, width=2)   # Beam
        self.canvas.create_line(200, 40, 200, 70, width=2)   # Rope

        total_parts = 6
        
        if incorrect_guesses_made > 0: # Head
            self.canvas.create_oval(180, 70, 220, 110, outline="black", width=2)
        if incorrect_guesses_made > (1 * self.initial_max_tries / total_parts): # Body
            self.canvas.create_line(200, 110, 200, 170, width=2)
        if incorrect_guesses_made > (2 * self.initial_max_tries / total_parts): # Left Arm
            self.canvas.create_line(200, 130, 170, 150, width=2)
        if incorrect_guesses_made > (3 * self.initial_max_tries / total_parts): # Right Arm
            self.canvas.create_line(200, 130, 230, 150, width=2)
        if incorrect_guesses_made > (4 * self.initial_max_tries / total_parts): # Left Leg
            self.canvas.create_line(200, 170, 170, 210, width=2)
        if incorrect_guesses_made > (5 * self.initial_max_tries / total_parts): # Right Leg
            self.canvas.create_line(200, 170, 230, 210, width=2)

    def make_guess_event(self, event):
        """Handles the 'Enter' key press event for guessing."""
        self.make_guess()

    def make_guess(self):
        """
        Processes the player's guess from the entry box.
        """
        guess = self.guess_entry.get().upper().strip()
        self.guess_entry.delete(0, tk.END)

        if not guess:
            messagebox.showwarning("Invalid Input", "Please enter a guess.")
            return

        if not guess.isalpha():
            messagebox.showwarning("Invalid Input", "Guess must contain only letters.")
            return
            
        if len(guess) == 1:
            if guess in self.guessed_letters:
                messagebox.showinfo("Already Guessed", f"You already guessed '{guess}'.")
            elif guess in self.secret_word:
                self.guessed_letters.add(guess)
                for i, char in enumerate(self.secret_word):
                    if char == guess:
                        self.word_display[i] = char
            else:
                self.guessed_letters.add(guess)
                self.current_tries -= 1
        
        elif len(guess) == len(self.secret_word):
            if guess in self.guessed_words:
                messagebox.showinfo("Already Guessed", f"You already guessed the word '{guess}'.")
            elif guess == self.secret_word:
                self.word_display = list(self.secret_word) 
            else: 
                self.guessed_words.add(guess)
                self.current_tries -= 1
        else:
            messagebox.showwarning("Invalid Input", "Please guess a single letter or the full word.")

        self.update_display()
        self.draw_hangman_figure()
        self.check_game_over()

    def get_hint(self):
        """
        Reveals one or more letters of the secret word as a hint.
        """
        if self.hints_remaining <= 0:
            messagebox.showinfo("No Hints Left", "You've used all your hints!")
            return

        unrevealed_letters = []
        for i, char in enumerate(self.secret_word):
            if self.word_display[i] == '_':
                unrevealed_letters.append(char)

        if not unrevealed_letters:
            messagebox.showinfo("No Hint Needed", "The word is almost revealed!")
            return

        num_to_reveal = min(len(set(unrevealed_letters)), self.letters_per_hint)
        letters_for_hint = random.sample(list(set(unrevealed_letters)), num_to_reveal)
        
        for hint_letter in letters_for_hint:
            if hint_letter not in self.guessed_letters:
                self.guessed_letters.add(hint_letter)
                for i, char in enumerate(self.secret_word):
                    if char == hint_letter:
                        self.word_display[i] = char
        
        self.hints_remaining -= 1
        if self.hints_remaining <= 0:
            self.hint_button.config(state=tk.DISABLED)

        self.update_display()
        self.check_game_over()

    def check_game_over(self):
        """
        Checks if the game has been won or lost and displays the result.
        """
        if "_" not in self.word_display:
            messagebox.showinfo("You Won!", f"Congratulations! You guessed the word: {self.secret_word}")
            self.disable_game_input()
        elif self.current_tries <= 0:
            messagebox.showinfo("Game Over", f"You ran out of tries! The word was: {self.secret_word}")
            self.disable_game_input()

    def disable_game_input(self):
        """
        Disables input fields and buttons at the end of a game.
        """
        self.guess_entry.config(state=tk.DISABLED)
        self.guess_button.config(state=tk.DISABLED)
        self.hint_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    game = HangmanGUI(root)
    root.mainloop()
