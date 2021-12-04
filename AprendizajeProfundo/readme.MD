
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

Las métricas del MLP (que puede encontrarse en ```experiment/mlp.py```) fueron las siguientes:
![alt text](https://github.com/ivanafeldfeber/diplo-datos-optativas/blob/main/AprendizajeProfundo/images/MLP.png?raw=true)
![alt text](https://github.com/ivanafeldfeber/diplo-datos-optativas/blob/main/AprendizajeProfundo/images/MLP%20graph.png?raw=true)

Como podemos ver en el gráfico la loss de entrenamiento y validación no dió muy bien, pero lo que vino después fue peor.

Intentamos tocar algunos hiperparámetros del Perceptron:
- learning rate =1e-4
- weight decay = 1e-4
- batch size = 1024

Y aprendió menos y peor:
![image](https://user-images.githubusercontent.com/8229279/144724582-9c8dbdc5-7ef2-42e3-b783-cbac902f7081.png)
![image](https://user-images.githubusercontent.com/8229279/144724589-29cfb03b-4925-4149-a6ce-90646efc9a1f.png)

Decidimos que necesitábamos darle un descanso al MLP por lo tanto probamos con una red neuronal convolucional (CNN) utilizando de base todo el código del Perceptrón y cambiando su arquitectura por una arquitectura de CNN sencilla que nos dieron en la materia para poder trabajar embeddings de análisis de sentimiento en IMDb.

Este modelo al principio no aprendía NADA, dandonos un balanced accuracy de 0.002. A partir de retocar hiperparámetros mejoró: empezamos con un learning rate muy alto 0.1 y finalmente funcionó mucho mejor con uno de 0.0005 y entrenando por 8 épocas en vez de 4 como veníamos haciendo. Su clase puede encontrarse en ```experiment/cnn1.py``` y su versión final tuvo las siguientes métricas:

![image](https://user-images.githubusercontent.com/8229279/144724616-9846f478-f904-48c5-ac32-573a3466f63c.png)
![image](https://user-images.githubusercontent.com/8229279/144724620-f563762d-290e-495b-aca9-62073616cc7b.png)

Mejor, pero todavía falta.

Luego de investigar un poco más nos dimos cuenta que el trabajo sobre los datos de IMDb tiene clases binarias, mientras que el dataset de Meli tiene como 600 clases, por lo tanto íbamos a necesitar una arquitectura un poco más compleja para que pueda aprender mejor de nuestros datos. 





