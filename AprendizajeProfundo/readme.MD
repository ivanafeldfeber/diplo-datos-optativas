
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

### MLP
Luego de un video muy explicativo de Cristian sobre cómo encolar trabajos comprenidmos la lógica detrás de Nabu2 y Jupyter Lab. Nos pusimos a armar y entrenar varios modelos, comenzando por el **Multi Layer Perceptron** que nos dieron como baseline desde la materia. Spoiler alert: ningún modelo nos dio muy bien, no hubiesemos ganado el Meli Challenge ni de casualidad.

Las métricas del MLP (que puede encontrarse en ```experiment/mlp.py```) fueron las siguientes:
![alt text](https://github.com/ivanafeldfeber/diplo-datos-optativas/blob/main/AprendizajeProfundo/images/MLP.png?raw=true)
![alt text](https://github.com/ivanafeldfeber/diplo-datos-optativas/blob/main/AprendizajeProfundo/images/MLP%20graph.png?raw=true)
![image](https://user-images.githubusercontent.com/8229279/144725030-ac834f05-347b-463c-95df-c5dc35144f4b.png)


Como podemos ver en el gráfico la loss de entrenamiento y validación no dió muy bien, aunque vemos que la balanced accuracy sigue creciendo, tal vez entrenarlo solo por 3 épocas no alcanzó.

### MLP con hiperparámetros 
Intentamos tocar algunos hiperparámetros del Perceptron, durante 10 épocas:
```
- learning rate =1e-4
- weight decay = 1e-4
- batch size = 1024
```

Y aprendió menos y peor:
![image](https://user-images.githubusercontent.com/8229279/144724582-9c8dbdc5-7ef2-42e3-b783-cbac902f7081.png)
![image](https://user-images.githubusercontent.com/8229279/144724589-29cfb03b-4925-4149-a6ce-90646efc9a1f.png)
![image](https://user-images.githubusercontent.com/8229279/144725037-6e082e0e-98e3-48f6-ba73-2ee6685b1e51.png)

Si bien la curva de BACC nos demuestra que está aprendiendo, lo hace muy lentamente y no son valores significativos.

### CNN 1
Decidimos que necesitábamos darle un descanso al MLP por lo tanto probamos con una red neuronal convolucional (CNN) utilizando de base todo el código del Perceptrón y cambiando su arquitectura por una arquitectura de CNN sencilla que nos dieron en la materia para poder trabajar embeddings de análisis de sentimiento en IMDb.

Este modelo al principio no aprendía NADA, dandonos un balanced accuracy de 0.002. A partir de retocar hiperparámetros mejoró: empezamos con un learning rate muy alto 0.1 y finalmente funcionó mucho mejor con uno de 0.0005 y entrenando por 8 épocas en vez de 4 como veníamos haciendo. Su clase puede encontrarse en ```experiment/cnn1.py``` y su versión final tuvo las siguientes métricas:

![image](https://user-images.githubusercontent.com/8229279/144724616-9846f478-f904-48c5-ac32-573a3466f63c.png)
![image](https://user-images.githubusercontent.com/8229279/144724620-f563762d-290e-495b-aca9-62073616cc7b.png)
![image](https://user-images.githubusercontent.com/8229279/144724988-357da59e-a0c6-44e9-8057-602a4cdc3e7b.png)


Mejor, pero todavía falta. Y no es un tema de épocas porque vemos al final como comienza a aplanarse la curva, ya no está mejorando su performance.

### CNN 2
Probamos con mismos hiperparámetros pero con otro optimizador, SGD, no con Adam pero directamente la red no aprendió nada. Investigamos el BCELoss, pero vimos que funciona mejor en clases binarias, entonces lo descartamos.

### CNN 3
Luego de investigar un poco más nos dimos cuenta que el trabajo sobre los datos de IMDb tiene clases binarias, mientras que el dataset de Meli tiene como 600 clases, por lo tanto íbamos a necesitar una arquitectura un poco más compleja para que pueda aprender mejor de nuestros datos. Agregamos capas ocultas y dropout en 0.3 y obtuvimos mejores resultados, entrenando al modelo por 10 épocas.  

![image](https://user-images.githubusercontent.com/8229279/144724764-ab75d8ba-6f25-47de-99a8-729d0a7bb6f1.png)
![image](https://user-images.githubusercontent.com/8229279/144724770-d4212676-2cce-4874-80ee-6114a70fb660.png)
![image](https://user-images.githubusercontent.com/8229279/144724877-0c5c6883-5ef5-49c6-9bde-06ce81c93b4e.png)



### Conclusiones, información y próximos pasos:
- En el archivo ```run.sh``` están los diferentes modelos que fuimos corriendo con diferentes hiperparámetros en cada modelo, lo hicimos de este modo, generando nuevos argumentos para que los archivos .py queden lo más "generales" posibles y que a la hora de ejecutar podamos elegir qué hiperparámetros usar.
- Es fundamental seguir mejorando nuestros modelos, creemos que tal vez con más épocas conseguiremos mejores resultados, sobre todo en la última Red Convolucional, que vemos que sigue subiendo la curva. Tal vez al final del día podamos subir mejores métricas, pero los deadlines son tiranos.
- Nos quedamos con ganas de seguir investigando otras arquitecturas y formas de entrenar estos modelos, queremos probar hacero fuera de Nabu2 así podemos replicarlo aunque no tengamos más acceso al servidor.


