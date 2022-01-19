import tkinter as tk
import methods as root_finder
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog as fd
from math import *


class RootCalculator:
    def __init__(self, master):
        # super().__init__(master)
        # self.self=master
        self.master = master
        self.myFrame = ttk.LabelFrame(self.master, text="Function")
        self.myFrame.pack(fill='both', side='left', anchor="nw", expand='false', padx=10, pady=10)
        self.methodFrame = ttk.LabelFrame(self.master, text="Choose Root Solving Method")
        self.methodFrame.pack(fill='both', side='top', expand='false', pady=10)
        self.tableFrame = ttk.LabelFrame(self.master, text="Table")
        self.tableFrame.pack(fill='both', side='bottom', anchor="sw", expand='true')
        self.iterationsTable = ttk.Treeview(self.tableFrame)
        self.resultLabel = ttk.Label(self.tableFrame)
        self.executionTimeLabel = ttk.Label(self.tableFrame)
        self.errorLabel = ttk.Label(self.tableFrame)
        self.createButton()
        self.createMethodSelector()

    def createMethodSelector(self):
        self.methodLabel = ttk.Label(self.methodFrame, text="Method")
        self.methodLabel.grid(row=0, column=0)
        self.options = ["Bisection", "False-Position", "Fixed Point", "Newton Raphson", "Secant"]
        self.clicked = tk.StringVar()
        self.clicked.set(self.options[0])
        self.clicked.trace("w", self.changeEntry)
        self.methodSelect = tk.OptionMenu(self.methodFrame, self.clicked, *self.options)
        self.methodSelect.grid(row=0, column=1, padx=10, pady=5)
        self.lowerBoundLabel = ttk.Label(self.methodFrame, text="x (lower bound) ")
        self.lowerBoundLabel.grid(row=1, column=0, padx=10, pady=3)
        self.lowerBoundEntry = ttk.Entry(self.methodFrame)
        self.lowerBoundEntry.grid(row=1, column=1, padx=10, pady=3)
        self.upperBoundLabel = ttk.Label(self.methodFrame, text="x (upper bound) ")
        self.upperBoundLabel.grid(row=1, column=2, padx=10, pady=3)
        self.upperBoundEntry = ttk.Entry(self.methodFrame)
        self.upperBoundEntry.grid(row=1, column=3, padx=10, pady=3)
        self.maxIterationsLabel = ttk.Label(self.methodFrame, text="Max Iterations ")
        self.maxIterationsLabel.grid(row=2, column=0, padx=10, pady=3)
        self.maxIterationsEntry = ttk.Entry(self.methodFrame)
        self.maxIterationsEntry.insert(0, "50")
        self.maxIterationsEntry.grid(row=2, column=1, padx=10, pady=3)
        self.toleranceLabel = ttk.Label(self.methodFrame, text="Tolerance ")
        self.toleranceLabel.grid(row=2, column=2, padx=10, pady=3)
        self.toleranceEntry = ttk.Entry(self.methodFrame)
        self.toleranceEntry.insert(0, "0.00001")
        self.toleranceEntry.grid(row=2, column=3, padx=10, pady=3)
        self.loadFileButton = ttk.Button(self.methodFrame, text="Load File", command=lambda: self.loadFile())
        self.loadFileButton.grid(row=1, column=6, padx=40, pady=20)
        self.singleMode = tk.IntVar()
        single_mode_check_box = ttk.Checkbutton(self.methodFrame, text="Single Mode", variable=self.singleMode)
        single_mode_check_box.grid(row=2, column=6, padx=40, pady=20)

    def changeEntry(self, *args):
        if self.clicked.get() == "Fixed Point" or self.clicked.get() == "Newton Raphson":
            self.upperBoundEntry.delete(0, tk.END)
            self.upperBoundEntry.insert(0, "0")
            self.upperBoundEntry.grid_forget()
            self.upperBoundLabel.grid_forget()
        else:
            self.upperBoundLabel.grid(row=1, column=2, padx=10, pady=5)
            self.upperBoundEntry.grid(row=1, column=3, padx=10, pady=5)

    def loadFile(self):
        file_name = fd.askopenfilename(title="Select A File", filetypes=[("txt files", ".txt")])
        file = open(file_name, 'r')
        file_input = []
        for line in file:
            file_input.append(line.rstrip("\n"))
        self.functionEntry.delete(0, tk.END)
        self.lowerBoundEntry.delete(0, tk.END)
        self.upperBoundEntry.delete(0, tk.END)
        if file_input[1] == "Bisection" or file_input[1] == "False-Position" or file_input[1] == "Secant":
            self.clicked.set(file_input[1])
            self.functionEntry.insert(0, file_input[0])
            self.lowerBoundEntry.insert(0, file_input[2])
            self.upperBoundEntry.insert(0, file_input[3])

            if len(file_input) == 6:
                self.maxIterationsEntry.delete(0, tk.END)
                self.maxIterationsEntry.insert(0, file_input[4])
                self.toleranceEntry.delete(0, tk.END)
                self.toleranceEntry.insert(0, file_input[5])
            else:
                self.functionEntry.insert(0, "Error Reading File")
        elif file_input[1] == "Fixed Point" or file_input[1] == "Newton Raphson":
            self.functionEntry.insert(0, file_input[0])
            self.clicked.set(file_input[1])
            self.lowerBoundEntry.insert(0, file_input[2])
            if len(file_input) == 5:
                self.maxIterationsEntry.delete(0, tk.END)
                self.maxIterationsEntry.insert(0, file_input[3])
                self.toleranceEntry.delete(0, tk.END)
                self.toleranceEntry.insert(0, file_input[4])
            else:
                self.functionEntry.insert(0, "Error Reading File")
        else:
            self.functionEntry.insert(0, "Error Reading File")
        file.close()

    def createButton(self):
        self.functionEntry = ttk.Entry(self.myFrame, width=50)
        self.functionEntry.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.functionEntry.event_generate('<Control-v>')
        self.functionEntry.event_generate('<Control-x>')
        self.functionEntry.event_generate('<Control-c>')

        self.button0 = ttk.Button(self.myFrame, text="0", command=lambda: self.buttonClick(0))
        self.button1 = ttk.Button(self.myFrame, text="1", command=lambda: self.buttonClick(1))
        self.button2 = ttk.Button(self.myFrame, text="2", command=lambda: self.buttonClick(2))
        self.button3 = ttk.Button(self.myFrame, text="3", command=lambda: self.buttonClick(3))
        self.button4 = ttk.Button(self.myFrame, text="4", command=lambda: self.buttonClick(4))
        self.button5 = ttk.Button(self.myFrame, text="5", command=lambda: self.buttonClick(5))
        self.button6 = ttk.Button(self.myFrame, text="6", command=lambda: self.buttonClick(6))
        self.button7 = ttk.Button(self.myFrame, text="7", command=lambda: self.buttonClick(7))
        self.button8 = ttk.Button(self.myFrame, text="8", command=lambda: self.buttonClick(8))
        self.button9 = ttk.Button(self.myFrame, text="9", command=lambda: self.buttonClick(9))

        self.buttonExp = ttk.Button(self.myFrame, text="exp", command=lambda: self.buttonClick("e("))
        self.buttonTan = ttk.Button(self.myFrame, text="tan", command=lambda: self.buttonClick("tan("))
        self.buttonSin = ttk.Button(self.myFrame, text="sin", command=lambda: self.buttonClick("sin("))
        self.buttonCos = ttk.Button(self.myFrame, text="cos", command=lambda: self.buttonClick("cos("))
        self.buttonLn = ttk.Button(self.myFrame, text="ln", command=lambda: self.buttonClick("ln("))
        self.buttonX = ttk.Button(self.myFrame, text="x", command=lambda: self.buttonClick("x"))

        self.buttonPlus = ttk.Button(self.myFrame, text="+", command=lambda: self.buttonClick("+"))
        self.buttonMult = ttk.Button(self.myFrame, text="*", command=lambda: self.buttonClick("*"))
        self.buttonDiv = ttk.Button(self.myFrame, text="/", command=lambda: self.buttonClick("/"))
        self.buttonMinus = ttk.Button(self.myFrame, text="-", command=lambda: self.buttonClick("-"))
        self.buttonPower = ttk.Button(self.myFrame, text="^", command=lambda: self.buttonClick("^("))
        self.buttonBracketLeft = ttk.Button(self.myFrame, text="(", command=lambda: self.buttonClick("("))
        self.buttonBracketRight = ttk.Button(self.myFrame, text=")", command=lambda: self.buttonClick(")"))

        self.buttonDel = ttk.Button(self.myFrame, text="Del", command=lambda: self.buttonDelete())
        self.buttonAC = ttk.Button(self.myFrame, text="AC", command=lambda: self.buttonClearClick())
        self.buttonEqual = ttk.Button(self.myFrame, text="Solve", command=lambda: self.buttonCalculate())

        self.buttonDel.grid(row=1, column=3, padx=5, pady=5)
        self.buttonAC.grid(row=0, column=3, padx=5, pady=5)
        self.buttonExp.grid(row=1, column=0, padx=5, pady=5)
        self.buttonLn.grid(row=1, column=1, padx=5, pady=5)
        self.buttonX.grid(row=1, column=2, padx=5, pady=5)
        self.buttonTan.grid(row=2, column=0, padx=5, pady=5)
        self.buttonSin.grid(row=2, column=1, padx=5, pady=5)
        self.buttonCos.grid(row=2, column=2, padx=5, pady=5)

        self.button7.grid(row=3, column=0, padx=5, pady=5)
        self.button8.grid(row=3, column=1, padx=5, pady=5)
        self.button9.grid(row=3, column=2, padx=5, pady=5)
        self.button4.grid(row=4, column=0, padx=5, pady=5)
        self.button5.grid(row=4, column=1, padx=5, pady=5)
        self.button6.grid(row=4, column=2, padx=5, pady=5)
        self.button1.grid(row=5, column=0, padx=5, pady=5)
        self.button2.grid(row=5, column=1, padx=5, pady=5)
        self.button3.grid(row=5, column=2, padx=5, pady=5)
        self.button0.grid(row=6, column=1, padx=5, pady=5)

        self.buttonBracketLeft.grid(row=6, column=0, padx=5, pady=5)
        self.buttonBracketRight.grid(row=6, column=2, padx=5, pady=5)
        self.buttonPower.grid(row=2, column=3, padx=5, pady=5)
        self.buttonDiv.grid(row=3, column=3, padx=5, pady=5)
        self.buttonMult.grid(row=4, column=3, padx=5, pady=5)
        self.buttonMinus.grid(row=5, column=3, padx=5, pady=5)
        self.buttonPlus.grid(row=6, column=3, padx=5, pady=5)
        self.buttonEqual.grid(row=7, column=0, columnspan=4, rowspan=2, padx=5, pady=5, sticky=tk.W + tk.E)

        self.readmeString = "Entry Restrictions:\n" \
                            "\tExponential denoted by e()\n" \
                            "\tMultiplication 4*x not 4x\n" \
                            "\tLogarithmic function denoted by ln()\n" \
                            "\tPowers denoted by ^\n" \
                            "\tSquare root denoted by ^(0.5)"
        self.readmeLabel = ttk.Label(self.myFrame, text=self.readmeString)
        self.readmeLabel.grid(row=9, column=0, columnspan=4, padx=10, pady=40)

    def buttonClick(self, op):
        self.functionEntry.insert(tk.END, op)

    def buttonClearClick(self):
        self.functionEntry.delete(0, tk.END)

    def buttonDelete(self):
        self.input = self.functionEntry.get()
        self.input = self.input[:-1]
        self.functionEntry.delete(0, tk.END)
        self.functionEntry.insert(0, self.input)

    def buttonCalculate(self):
        plt.close("all")
        self.errorLabel['text'] = ""
        self.errorLabel.grid_forget()
        self.input = self.functionEntry.get()
        try:
            answer = root_finder.selectMethod(str(self.clicked.get()), str(self.input),
                                              float(self.lowerBoundEntry.get()), float(self.upperBoundEntry.get()),
                                              float(self.toleranceEntry.get()), int(self.maxIterationsEntry.get()),
                                              int(self.singleMode.get()))
            method_chosen = str(self.clicked.get())
            if type(answer) != str:
                if method_chosen == "Bisection" or method_chosen == "False-Position":
                    self.createTable(answer)
                elif method_chosen == "Newton Raphson":
                    self.newtonRaphsonTable(answer)
                elif method_chosen == "Secant":
                    self.secantTable(answer)
                elif method_chosen == "Fixed Point":
                    self.fixedPointTable(answer)
            else:
                self.executionTimeLabel['text'] = ""
                self.resultLabel['text'] = ""
                for item in self.iterationsTable.get_children():
                    self.iterationsTable.delete(item)
                self.errorString = answer
                self.errorLabel['text'] = self.errorString
                self.errorLabel.pack(padx=20, pady=(20, 0))
        except Exception as e:
            if "zero" in str(e):
                self.errorLabel['text'] = "Error: Division by Zero !"
            elif "float" in str(e):
                self.errorLabel['text'] = "Error: Field Left Empty !"
            elif "invalid" in str(e):
                self.errorLabel['text'] = "Error: Invalid Input"
            else:
                self.errorLabel['text'] = str(e)
            self.executionTimeLabel.pack_forget()
            self.resultLabel.pack_forget()
            self.iterationsTable.pack_forget()
            self.errorLabel.pack(padx=20, pady=(20, 0))

    def plotFunction(self):
        x_list = np.linspace(-10, 10, num=1000)
        plt.figure(num=0, dpi=120)
        plt.plot(x_list, self.func(x_list))
        plt.title("f(x)")
        plt.xlabel("x")
        plt.grid()
        plt.ylabel("y")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()

    def func(self, x):
        expression = self.input.replace("^", "**")
        expression = expression.replace("cos", "np.cos")
        expression = expression.replace("sin", "np.sin")
        expression = expression.replace("tan", "np.tan")
        expression = expression.replace("ln", "np.log")
        expression = expression.replace("exp", "np.exp")
        return eval(expression)

    def createTable(self, result):
        data = result['table']
        self.executionTimeLabel['text'] = ""
        self.resultLabel['text'] = ""
        for item in self.iterationsTable.get_children():
            self.iterationsTable.delete(item)
        self.iterationsTable['columns'] = ("i", "Xl", "Xu", "Xr", "f(Xr)", "Relative Error")
        self.iterationsTable.column("#0", width=0, stretch=tk.NO)
        self.iterationsTable.column("i", anchor=tk.W, width=30)
        self.iterationsTable.column("Xl", anchor=tk.W, width=150)
        self.iterationsTable.column("Xu", anchor=tk.W, width=150)
        self.iterationsTable.column("Xr", anchor=tk.W, width=150)
        self.iterationsTable.column("f(Xr)", anchor=tk.W, width=150)
        self.iterationsTable.column("Relative Error", anchor=tk.W, width=150)

        self.iterationsTable.heading("i", text="i", anchor=tk.W)
        self.iterationsTable.heading("Xl", text="Xl", anchor=tk.W)
        self.iterationsTable.heading("Xu", text="Xu", anchor=tk.W)
        self.iterationsTable.heading("Xr", text="Xr", anchor=tk.W)
        self.iterationsTable.heading("f(Xr)", text="f(Xr)", anchor=tk.W)
        self.iterationsTable.heading("Relative Error", text="Relative Error", anchor=tk.W)

        count = 0
        for record in data:
            self.iterationsTable.insert(parent="", index="end", iid=count, text="",
                                        values=(record[0], record[1], record[2], record[3], record[4], record[5]))
            count += 1
        self.iterationsTable.pack(padx=20, pady=20)
        self.resultString = "Root x at i=" + str(count) + ":\t" + str(data[-1][3])
        self.resultLabel['text'] = self.resultString
        self.resultLabel.pack(padx=20, pady=10)
        self.executionTimeString = "Execution Time = " + str(result["execution_time"]) + "s"
        self.executionTimeLabel['text'] = self.executionTimeString
        self.executionTimeLabel.pack(padx=20, pady=10)

    def newtonRaphsonTable(self, result):
        data = result['table']
        self.executionTimeLabel['text'] = ""
        self.resultLabel['text'] = ""
        for item in self.iterationsTable.get_children():
            self.iterationsTable.delete(item)
        self.iterationsTable['columns'] = ("i", "Xold", "Xnew", "f(x)", "f`(x)", "Relative Error")
        self.iterationsTable.column("#0", width=0, stretch=tk.NO)
        self.iterationsTable.column("i", anchor=tk.W, width=30)
        self.iterationsTable.column("Xold", anchor=tk.W, width=150)
        self.iterationsTable.column("Xnew", anchor=tk.W, width=150)
        self.iterationsTable.column("f(x)", anchor=tk.W, width=150)
        self.iterationsTable.column("f`(x)", anchor=tk.W, width=150)
        self.iterationsTable.column("Relative Error", anchor=tk.W, width=150)

        self.iterationsTable.heading("i", text="i", anchor=tk.W)
        self.iterationsTable.heading("Xold", text="Xold", anchor=tk.W)
        self.iterationsTable.heading("Xnew", text="Xnew", anchor=tk.W)
        self.iterationsTable.heading("f(x)", text="f(x)", anchor=tk.W)
        self.iterationsTable.heading("f`(x)", text="f`(x)", anchor=tk.W)
        self.iterationsTable.heading("Relative Error", text="Relative Error", anchor=tk.W)

        count = 0
        for record in data:
            self.iterationsTable.insert(parent="", index="end", iid=count, text="",
                                        values=(record[0], record[1], record[2], record[3], record[4], record[5]))
            count += 1
        self.iterationsTable.pack(padx=20, pady=20)
        self.resultString = "Root x at i=" + str(count) + ":\t" + str(data[-1][2])
        self.resultLabel['text'] = self.resultString
        self.resultLabel.pack(padx=20, pady=10)
        self.executionTimeString = "Execution Time = " + str(result["execution_time"]) + "s"
        self.executionTimeLabel['text'] = self.executionTimeString
        self.executionTimeLabel.pack(padx=20, pady=10)

    def secantTable(self, result):
        data = result['table']
        self.executionTimeLabel['text'] = ""
        self.resultLabel['text'] = ""
        for item in self.iterationsTable.get_children():
            self.iterationsTable.delete(item)
        self.iterationsTable['columns'] = ("i", "Xi-1", "Xi", "f(Xi-1)", "f(Xi)", "Xi+1", "Relative Error")
        self.iterationsTable.column("#0", width=0, stretch=tk.NO)
        self.iterationsTable.column("i", anchor=tk.W, width=20)
        self.iterationsTable.column("Xi-1", anchor=tk.W, width=130)
        self.iterationsTable.column("Xi", anchor=tk.W, width=130)
        self.iterationsTable.column("f(Xi-1)", anchor=tk.W, width=130)
        self.iterationsTable.column("f(Xi)", anchor=tk.W, width=130)
        self.iterationsTable.column("Xi+1", anchor=tk.W, width=130)
        self.iterationsTable.column("Relative Error", anchor=tk.W, width=130)

        self.iterationsTable.heading("i", text="i", anchor=tk.W)
        self.iterationsTable.heading("Xi-1", text="Xi-1", anchor=tk.W)
        self.iterationsTable.heading("Xi", text="Xi", anchor=tk.W)
        self.iterationsTable.heading("f(Xi-1)", text="f(Xi-1)", anchor=tk.W)
        self.iterationsTable.heading("f(Xi)", text="f(Xi)", anchor=tk.W)
        self.iterationsTable.heading("Xi+1", text="Xi+1", anchor=tk.W)
        self.iterationsTable.heading("Relative Error", text="Relative Error", anchor=tk.W)

        count = 0
        for record in data:
            self.iterationsTable.insert(parent="", index="end", iid=count, text="",
                                        values=(
                                            record[0], record[1], record[2], record[3], record[4], record[5],
                                            record[6]))
            count += 1
        self.iterationsTable.pack(padx=20, pady=20)
        self.resultString = "Root x at i=" + str(count) + ":\t" + str(data[-1][5])
        self.resultLabel['text'] = self.resultString
        self.resultLabel.pack(padx=20, pady=10)
        self.executionTimeString = "Execution Time = " + str(result["execution_time"]) + "s"
        self.executionTimeLabel['text'] = self.executionTimeString
        self.executionTimeLabel.pack(padx=20, pady=10)

    def fixedPointTable(self, result):
        data = result['table']
        self.executionTimeLabel['text'] = ""
        self.resultLabel['text'] = ""
        for item in self.iterationsTable.get_children():
            self.iterationsTable.delete(item)
        self.iterationsTable['columns'] = ("i", "Xi", "Xi+1", "Relative Error")
        self.iterationsTable.column("#0", width=0, stretch=tk.NO)
        self.iterationsTable.column("i", anchor=tk.W, width=20)
        self.iterationsTable.column("Xi", anchor=tk.W, width=130)
        self.iterationsTable.column("Xi+1", anchor=tk.W, width=130)
        self.iterationsTable.column("Relative Error", anchor=tk.W, width=130)

        self.iterationsTable.heading("i", text="i", anchor=tk.W)
        self.iterationsTable.heading("Xi", text="Xi", anchor=tk.W)
        self.iterationsTable.heading("Xi+1", text="Xi+1", anchor=tk.W)
        self.iterationsTable.heading("Relative Error", text="Relative Error", anchor=tk.W)

        count = 0
        for record in data:
            self.iterationsTable.insert(parent="", index="end", iid=count, text="",
                                        values=(record[0], record[1], record[2], record[3]))
            count += 1
        self.iterationsTable.pack(padx=20, pady=20)
        self.resultString = "Root x at i=" + str(count) + ":\t" + str(data[-1][2])
        self.resultLabel['text'] = self.resultString
        self.resultLabel.pack(padx=20, pady=10)
        self.executionTimeString = "Execution Time = " + str(result["execution_time"]) + "s"
        self.executionTimeLabel['text'] = self.executionTimeString
        self.executionTimeLabel.pack(padx=20, pady=10)


root = tk.Tk()
root.title("Root Calculator")
root.geometry('1300x700')
s = ttk.Style()
s.theme_use('xpnative')
rootCalculator = RootCalculator(root)
root.mainloop()
