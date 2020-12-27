from Classes.Car import Car
from flask import Flask, render_template, redirect, request


app = Flask(__name__)
car = Car()

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/mode', methods=['POST'])
def mode():
	if request.method == 'POST':
		if request.form['type'] == 'Manual':
			return render_template('manual.html')
		if request.form['type'] == 'Autonomous':
			return render_template('autonomous.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
