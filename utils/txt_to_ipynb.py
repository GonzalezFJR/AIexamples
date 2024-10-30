import nbformat
import re

def txt_to_ipynb(input_file, output_file):
    # Lee el archivo de entrada
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Inicializa la estructura del notebook
    nb = nbformat.v4.new_notebook()
    current_cell = []
    is_code = False
    
    # Procesa cada línea
    for line in lines:
        # Detecta el inicio o fin de una celda de código
        if line.strip() == "```python":
            # Guarda la celda actual como Markdown si no es vacía
            if current_cell and not is_code:
                cell_content = ''.join(current_cell)
                # Reemplaza delimitadores de fórmulas de \[ \] a $$ $$
                cell_content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', cell_content, flags=re.DOTALL)
                nb.cells.append(nbformat.v4.new_markdown_cell(cell_content))
            current_cell = []
            is_code = True
        elif line.strip() == "```":
            # Guarda la celda actual como código si no es vacía
            if current_cell and is_code:
                nb.cells.append(nbformat.v4.new_code_cell(''.join(current_cell)))
            current_cell = []
            is_code = False
        else:
            # Añade líneas a la celda actual
            current_cell.append(line)
    
    # Guarda cualquier celda restante
    if current_cell:
        cell_content = ''.join(current_cell)
        if is_code:
            nb.cells.append(nbformat.v4.new_code_cell(cell_content))
        else:
            # Reemplaza delimitadores de fórmulas de \[ \] a $$ $$
            cell_content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', cell_content, flags=re.DOTALL)
            nb.cells.append(nbformat.v4.new_markdown_cell(cell_content))
    
    # Guarda el notebook en formato .ipynb
    with open(output_file, 'w', encoding='utf-8') as file:
        nbformat.write(nb, file)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convierte un archivo de texto en un notebook de Jupyter.')
    parser.add_argument('fname', type=str, help='Nombre del archivo de entrada.')
    args = parser.parse_args()
    oname = args.fname.split('.')[0] + '.ipynb'
    txt_to_ipynb(args.fname, oname)