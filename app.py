from flask import Flask, render_template, request, jsonify
import modules.customer_network as cn
import modules.disk_based_tree as dt
import modules.marketing_task as mt
import modules.product_search as ps
import modules.system_test as st
import modules.visualization_utils as vu

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_module():
    module = request.json.get('module')
    result = ""
    try:
        if module == 'customer_network':
            result = cn.run_network() if hasattr(cn, 'run_network') else "No 'run_network' function found."
        elif module == 'disk_based_tree':
            result = dt.build_tree() if hasattr(dt, 'build_tree') else "No 'build_tree' function found."
        elif module == 'marketing_task':
            result = mt.run_marketing() if hasattr(mt, 'run_marketing') else "No 'run_marketing' function found."
        elif module == 'product_search':
            result = ps.search_product() if hasattr(ps, 'search_product') else "No 'search_product' function found."
        elif module == 'system_test':
            result = st.run_test() if hasattr(st, 'run_test') else "No 'run_test' function found."
        elif module == 'visualization_utils':
            result = vu.generate_plot() if hasattr(vu, 'generate_plot') else "No 'generate_plot' function found."
        else:
            result = "Invalid module."
    except Exception as e:
        result = f"Error running module: {str(e)}"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)