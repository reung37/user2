#include <iostream>
#include <cstdlib>
#include <stack>
#include <cstring>
#include <cctype>

using namespace std;

void evalutate_stack(stack<double>& numbers, stack<char>& operations) {
	double operand1, operand2;
	operand2 = numbers.top();
	numbers.pop();
	operand1 = numbers.top();
	numbers.pop();

	switch (operations.top()) {
	case '+': numbers.push(operand1 + operand2);
		break;
	case '-': numbers.push(operand1 - operand2 + operand2);
		break;
	case '*': numbers.push(operand1 * operand2);
		break;
	case '/': 
		if (operand2 == 0) {
			cout << "Error!: Divide by zero exception" <<endl;
			break;
		}
		else
			numbers.push(operand1 / operand2);
		break;
	}
	operations.pop();
}

double read_and_evaluate(istream &ins) {
	const char DECIMAL = '.';
	const char LEFT_PARENTHESIS = '(';
	const char L_CURLY_BRACKETS = '{';
	const char L_SQUARE_BRACKETS = '[';

	const char RIGHT_PARENTHESIS = ')';
	const char CURLY_BRACKETS = '}';
	const char SQUARE_BRACKETS = ']';

	stack<double> numbers;
	stack<char> operations;
	double number;
	char symbol;

	while (ins && ins.peek() != '\n')
	{
		if (isdigit(ins.peek()) || ins.peek() == DECIMAL) {
			ins >> number;
			numbers.push(number);
			if (operations.size() != 0) {
				if (operations.top() == '*' || operations.top() == '/')
					evalutate_stack(numbers, operations);
			}
		}
		else if (strchr("+-*/", ins.peek()) != NULL)
		{
			ins >> symbol;
			if (symbol == '+' || symbol == '-') {
				if (operations.size() != 0) {
					if (operations.top() == '+' || operations.top() == '-')
						evalutate_stack(numbers, operations);
				}
			}
			operations.push(symbol);
		}
		
		else if (ins.peek() == LEFT_PARENTHESIS) {
			ins >> symbol;
			operations.push(symbol);
		}
		else if (ins.peek() == L_CURLY_BRACKETS) {
			ins >> symbol;
			operations.push(symbol);
		}
		else if (ins.peek() == L_SQUARE_BRACKETS) {
			ins >> symbol;
			operations.push(symbol);
		}


		else if (ins.peek() == RIGHT_PARENTHESIS) {
			ins.ignore();
			while (1) {
				if (operations.size() != 0) {
					if (operations.top() == LEFT_PARENTHESIS) {
						operations.pop();
						break;
					}
					else if (numbers.size() == 1 && strchr(("{["), operations.top())) {
						cout << "Error!: unbalanced parentheses" << endl;
						return -2;
					}
				}
				else {
					cout << "Error!: unbalanced parentheses" << endl;
					return -2;
				}
				evalutate_stack(numbers, operations);
			}
		}
		else if (ins.peek() == CURLY_BRACKETS) {
			ins.ignore();
			while (1) {
				if (operations.size() != 0) {
					if (operations.top() == L_CURLY_BRACKETS) {
						operations.pop();
						break;
					}
					else if (numbers.size() == 1 && strchr(("(["), operations.top())) {
						cout << "Error!: unbalanced parentheses" << endl;
						return -2;
					}
				}
				else {
					cout << "Error!: unbalanced parentheses" << endl;
					return -2;
				}
				evalutate_stack(numbers, operations);
			}
		}
		else if (ins.peek() == SQUARE_BRACKETS) {
			ins.ignore();
			while (1) {
				if (operations.size() != 0) {
					if (operations.top() == L_SQUARE_BRACKETS) {
						operations.pop();
						break;
					}
					else if (numbers.size() == 1 && strchr(("({"), operations.top())) {
						cout << "Error!: unbalanced parentheses" << endl;
						return -2;
					}
				}
				else {
					cout << "Error!: unbalanced parentheses" << endl;
					return -2;
				}
				evalutate_stack(numbers, operations);
			}
		}


		else if (ins.peek() == 'E') {
			ins.get();
			if (ins.peek() == 'O') {
				ins.get();
				if (ins.peek() == 'I') {
					return -1;
				}
			}
		}
		else {
			ins.ignore();
		}
	}
	while (operations.size() != 0) {
		if (operations.size() != 0) {
			if (operations.top() == LEFT_PARENTHESIS) {
				cout << "Error!: unbalanced parentheses" << endl;
				return -2;
			}
			if (operations.top() == L_CURLY_BRACKETS) {
				cout << "Error!: unbalanced parentheses" << endl;
				return -2;
			}
			if (operations.top() == L_SQUARE_BRACKETS) {
				cout << "Error!: unbalanced parentheses" << endl;
				return -2;
			}
		}
		evalutate_stack(numbers, operations);
	}
	if (!numbers.empty())
		return numbers.top();
	else
		return -2;
}

int main(void) 
{
	while (1)
	{
		double answer;

		answer = read_and_evaluate(cin);
		if (answer == -1)
			break;
		if (answer == -2) {
			cin.ignore(100, '\n');
			continue;
		}
		cout.setf(ios::fixed);
		cout.precision(3);
		cout << answer << endl;
		cin.ignore(1);
	}
}