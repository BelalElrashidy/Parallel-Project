from flask import Flask, render_template, request, jsonify
import subprocess
from flask import Flask, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def run_mpi_task(script_name,np, args=[]):
    # Run your MPI script with subprocess
    cmd = ["mpirun", "-np", str(np), "python3", "mpi_tasks/tasks.py", script_name] + args
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/another')
def another():
    return render_template('select.html')


@app.route('/upload_two_files', methods=['POST'])
def upload_two_files():
    args = []

    # Handle first file
    
    file1 = request.files['file1']
    if file1 :
        filename1 = secure_filename(file1.filename)
        path1 = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename1}")
        file1.save(path1)
        args.append(path1)

    # Handle second file
    
    file2 = request.files['file2']

    if file2:
        
        filename2 = secure_filename(file2.filename)
        path2 = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{filename2}")
        file2.save(path2)
        args.append(path2)
    
    data = request.form.get('extra')

    
    task = request.form.get('task')
    np = request.form.get('np', '4')
    if data:
        if task == "sort":
            
            data = str(data).split(' ')
            print(data)
            Data = [float(s) for s in data]
            # Data = np.array(Data)
            args.append(Data)

        else:
            args.append(data)
    if not task:
        return jsonify({"error": "Task not specified"}), 400

    # You can add task validation here if you want

    stdout, stderr = run_mpi_task(task, np, args)

    if stderr:
        return jsonify({"error": stderr}), 500
    return jsonify({"output": stdout})
@app.route('/run_task', methods=['POST'])
def run_task():
    task = request.form.get('task')
    args = request.form.getlist('args[]')  # optional args
    num = request.form.get("np",'4')

    if task == "matrix_multiply":
        stdout, stderr = run_mpi_task(task,num, args)
    elif task == "parallel_search":
        stdout, stderr = run_mpi_task(task,num, args)
    elif task == "linear_regression":
        stdout, stderr = run_mpi_task(task,num, args)
    elif task == "file_process":
        stdout, stderr = run_mpi_task(task,num, args)
    elif task == "image_process":
        stdout, stderr = run_mpi_task(task,num, args)
    elif task == "sort":
        stdout, stderr = run_mpi_task(task,num, args)
    elif task == "statics-ana":
        stdout, stderr = run_mpi_task(task,num, args)
    else:
        return jsonify({"error": "Unknown task"}), 400

    if stderr:
        return jsonify({"error": stderr}), 500
    return jsonify({"output": stdout})

if __name__ == "__main__":
    app.run(debug=True)
