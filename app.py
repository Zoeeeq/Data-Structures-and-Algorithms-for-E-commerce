from flask import Flask, render_template, request, jsonify
import importlib
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_module():
    data = request.json
    module_name = data.get('module')
    parameters = data.get('parameters', {})

    try:
        module = importlib.import_module(f"modules.{module_name}")
        if hasattr(module, 'run'):
            result = module.run(parameters)
        else:
            result = f"No 'run' function found in module '{module_name}'."
    except Exception as e:
        result = f"Error executing module '{module_name}': {str(e)}"

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
