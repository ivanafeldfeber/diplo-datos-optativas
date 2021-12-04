
# Aprendizaje Profundo
## Meli Challenge 2019
### Breve introducción
Este práctico tuvo como objetivo familiarizarnos con diversos modelos de Aprendizaje Profundo, entrenándolos con los datos del Meli Challenge 2019.
Si bien durante la materia pudimos ver cómo se preprocesan los datos y cómo se utilizan los embeddings, a fines de este trabajo utilizamos los datos ya preprocesados que nos dio el equipo docente.

### El proceso
Este grupo de trabajo está conformado por especialistas de diversas disciplinas, pero ninguna persona es de ingeniería o computación.
Somos:
- [Ivana Feldfeber](https://github.com/ivanafeldfeber)
- [Eduardo Barseghian](https://github.com/EduBarseghian)
- [Susana Araujo](https://github.com/suaraujo)
- Tamara Maggioni

A pesar de estos pequeños detalles con respecto a nuestras formaciones, elegimos ir por el camino difícil, el camino del computólogo. No vamos a negar que hubo momentos en los que pensamos ¿y porqué no hacemos todo en una sola Jupyter Notebook en Colab como venimos haciendo en vez de meternos en _el temible servidor_ de FAMAF denominado Nabuconodosor2? (sí, como la nave de Morfeo)

![alt text](https://i.imgflip.com/5wn3wz.jpg) 

Luego de un video muy explicativo de Cristian sobre cómo encolar trabajos comprenidmos la lógica detrás de Nabu2 y Jupyter Lab. Nos pusimos a armar y entrenar varios modelos, comenzando por el **Multi Layer Perceptron** que nos dieron como baseline desde la materia. Spoiler alert: ningún modelo nos dio muy bien, no hubiesemos ganado el Meli Challenge ni de casualidad.

Las métricas del MLP (que puede encontrarse en experiment/mlp.py) fueron las siguientes:
![alt text](https://github.com/ivanafeldfeber/diplo-datos-optativas/blob/main/AprendizajeProfundo/images/MLP.png?raw=true)
![alt text](https://github.com/ivanafeldfeber/diplo-datos-optativas/blob/main/AprendizajeProfundo/images/MLP%20graph.png?raw=true)


