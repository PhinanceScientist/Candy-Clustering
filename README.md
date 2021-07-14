<p align="center">
<a href="https://github.com/PhinanceScientist"><img src = "https://raw.githubusercontent.com/PhinanceScientist/PhinanceScientist/main/assets/ln-logo.png" width = 100> </a>
</p>
<h1 align=center><font size = 5>Machine Learning en la industria del retail.</font></h1>
<br>

# Introducción
<br>
Este proyecto fue basado en un ejercicio usado de ejemplo durante el curso de <a href="https://platzi.com/clases/scikitlearn-ml/">Scikiti-Learn</a> en <a href="https://platzi.com/">Platzi</a>.<br><br>

Para este ejercicio estaré usando el algoritmo de agrupación llamado "K-means" para resolver la tarea de agrupar 85 distintos dulces basándonos en sus propiedades, en este caso, basándonos en los valores de las columnas dentro del set de datos.<br><br>

Agradecimiento especial al profesor <a href="https://www.linkedin.com/in/arielortiz/">Ariel Ortiz</a> por la excelente clase.



***
# Librerias
<br>
  
   <li> <b>joblib:</b>   1.0.1<br></li>
   <li> <b>numpy </b> 1.21.0<br></li>
   <li> <b>pandas:</b> 1.3.0<br></li>   
   <li> <b>scikit-learn:</b> 0.24.2<br></li>
   <li> <b>scipy:</b> 1.7.0 <br></li>
  <br>





***
# Justificación del proyecto
<br>


En la industria del Retail, o Comercio Minorista, es uno de los mercados más interesantes y con mayores cambios en estos tiempos de pandemia.

Durante el 2019 el valor global de ésta industria llegó a los $25 Billones de dólares ($25 Trillion Dollars), de los cuales más del 75% fue generado por tiendas físicas. Se pronostica que para el 2022 el valor total de la industria del retail llegue a los $27 Billones ($27 Trillion Dollars) de dólares.

Una de las características más notables que podemos encontrar en esta industria se puede resumir bajo la siguiente frase: "**Los mejores productos, sin una disposición adecuada, no se venden**." Esta frase hace referencia a distintos problemas relacionados con el comportamientos de los compradores y la relación que tenga la organización de los productos durante la experiencia de compra. 

Si bien la manera en que acomodamos los productos influyen en el éxito de venta de los productos, también la oferta de opciones de productos similares pude considerarse un factor sumamente importante para analizar.

Existe un sinfín de teoría muy interesante sobre la forma en que las personas toman decisiones, te recomiendo <a href="https://www.youtube.com/watch?v=VO6XEQIsCoM&ab_channel=TED">esta plática</a> de Barry Schwartz en donde nos habla acerca de "La paradoja de la elección y porqué menos es más" o <a href="https://www0.gsb.columbia.edu/mygsb/faculty/research/pubfiles/5228/better%20choosing%20experience.pdf">este muy interesante artículo </a> de Sheena Lyengar y Kanika Agrawal llamado "Una mejor experiencia de elección" donde nos cuentan cómo la percepción de los productos afecta en la decisión final del consumidor.

Ambos artículos pueden resumir una misma idea, el consumidor quiere más opciones pero no demasiadas. Tener un amplio catálogo para escoger puede generar "parálisis de compra", esto es,  generar una mayor sensación de libertar para el consumidor pero con un porcentaje de ventas exitosas mucho menor debido a la abrumadora oferta y el extenuante proceso de elección que debe realizarse para tomar una decisión.

Podemos encontrar diferentes maneras de reducir estos inconvenientes, sin embargo, una de las recomendaciones de nuestros autores ofrece una forma de disminuir el llamado "Parálisis", pensar por el cliente. ¿Cómo logramos facilitar esta tarea al consumidor?, agrupando productos.

Un estudio realizado a los supermercados Wegmans demostró resultados fascinantes experimentando con la agrupación de revistas. En las sucursales donde las revistas estaban organizadas por categorías como "Hogar", "Deportes" y demás, los consumidores reportaron tener una experiencia de compra más placentera mencionando que sentían mayor oferta de variedad.  Esto contrasta con los resultados de las sucursales que no categorizaron las revistas incluso teniendo el mismo número de ejemplares.

Como podemos observar, la categorización adecuada de los productos puede generar una mayor conversión de ventas al impactar directamente en la experiencia del consumidor.

Por suerte para nosotros, existe un algoritmo de categorización llamado K-means que nos puede facilitar el trabajo de la creación de grupos de productos basado en las propiedades que comparten. Esto no es más que aplicar unas instrucciones matemáticas a una base de datos que incluya las características de nuestros productos. Con esta información podremos crear y solicitar un listado de aquellos productos miembros de cada grupo en un intento de mejorar la experiencia del consumidor y garantizar mayores conversiones de ventas.

Para este proyecto utilizaremos la base de datos llamada The Ultimate Halloween Candy Power Ranking creada por FiveThirtyEight. Este conjunto de datos fue recolectado al crear una página web en donde a los participantes se les mostraba dos diferentes dulces y se les preguntaba su favorito. Más de 269,000 votos fueron realizados de 8,371 diferentes direcciones de IP. (FiveThirtyEight, 2017)

Cada columna representa una característica del dulce y se utiliza el valor de "1" en caso de cumplir con la característica o "0" en caso de no cumplirla. 3 columnas fueron creadas adicionalmente con los resultados del experimento.

Las características de la colección de datos son las siguientes:

- chocolate: ¿Contiene chocolate?
- fruity: ¿Tiene sabor frutal?
- caramel: ¿Contiene caramelo?
- peanutalmondy: ¿Contiene cacahuates, crema de cacahuates, o almendras?
- nougat: ¿Contiene nougat?
- crispedricewafer: ¿Contiene arroz inflado, obleas o componentes de galletas?
- hard: ¿Es un dulce macizo?
- bar: ¿Es un dulce en barra?
- pluribus: ¿Es uno de muchos dulces de una misma bolsa o caja?
- sugarpercent: El parcentaje de contenido de azúcar con relación al resto de dulces.
- pricepercent: La unidad de precio con relación al resto de dulces.
- winpercent: Porcentaje de victorias acorde a 269,000 comparaciones.

Usaremos Python como nuestro lenguaje junto con Pandas y  Scikit-Learn como nuestras herramientas para ejecutar el algoritmo en nuestros datos y obtener como resultado nuestros grupos en un formato de marco de datos que faciliten su interpretación.

Usar Sckit-Learn es extremadamente sencillo si sabemos cómo definir los parámetros de nuestro algoritmo.

En pocas palabras solo tenemos que seguir los siguientes pasos:

 seleccionar nuestro documento de entrada (input), seleccionar el nombre de la columna que no utilizaremos

- Importar las librerías que necesitaremos
- Seleccionar nuestro documento de entrada
- Seleccionar el nombre de la columna que no usaremos. El nombre del dulce se descarta para el análisis al ser irrelevante para el algoritmo
- Definir el número de grupos con el parámetro "n_clusters", en este caso definimos 4.
- Incrustar nuestra predicción en un nuevo set de datos usando una columna con el nombre "grupo".
- Incrustar la columna del nombre del dulce.
- Exportar a un archivo en la carpeta deseada.

Adicional he agregado un par de líneas de código para poder observar los grupos utilizando la terminal al ejecutar el archivo de Python. Este paso es opcional pero nos brinda información ordenada por grupos de forma inmediata sin tener que abrir el archivo. Por favor considera que se trata de 4 grupos distintos iniciando desde el número 0.


De esta forma podemos estar seguros que nuestros dulces comparten características que los hacen pertenecer a un mismo grupo. Esta información podría ser de utilidad para la toma de decisiones al momento de planificar la disposición de productos.

Este es un claro ejemplo de cómo podemos agilizar un proceso utilizando las tecnologías para la agrupación de productos utilizando el algoritmo de K-means en Python.


***
 # Conclusión<br>
En conclusión, podemos utilizar las tecnologías para facilitar nuestro trabajo pero siempre debemos tener cuidado con la toma de decisiones, en una disciplina tan amplia como la mercadotecnia para el retail siempre es necesaria la intervención de personas expertas para mejorar la información que tenemos incluida la obtenida por tecnologías. Este experimento se trata simplemente de una demostración sobre el potencial latente que existe en las tecnologías dentro de las industrias modernas.

¿Te ha gustado este tema? Soy un fanático de las tecnologías y su aplicación casos prácticos de negocios. Si quieres agregar información adicional o incluso contar tu punto de vista, no dudes en contactarme o dejarme un comentario aquí mismo. Te agradezco tu tiempo y espero que hayas disfrutado de este ejercicio tanto como yo al realizarlo.

<br>
¡Que estés bien!

 
 ***

 ## Bibliografy
[https://www.sciencedirect.com/science/article/pii/S1877050915035929](https://www.sciencedirect.com/science/article/pii/S1877050915035929)

[https://courses.lumenlearning.com/wm-retailmanagement/chapter/layout-of-products/](https://courses.lumenlearning.com/wm-retailmanagement/chapter/layout-of-products/)

[https://www.zoho.com/inventory/guides/what-is-item-grouping.html](https://www.zoho.com/inventory/guides/what-is-item-grouping.html)

[https://erply.com/why-the-right-product-selection-is-crucially-important-for-a-retailer/](https://erply.com/why-the-right-product-selection-is-crucially-important-for-a-retailer/)

[https://www.youtube.com/watch?v=Y4mtHkCilxw&ab_channel=OmarDariasConde](https://www.youtube.com/watch?v=Y4mtHkCilxw&ab_channel=OmarDariasConde)

[https://www.youtube.com/watch?v=VO6XEQIsCoM&ab_channel=TED](https://www.youtube.com/watch?v=VO6XEQIsCoM&ab_channel=TED)

[https://www0.gsb.columbia.edu/mygsb/faculty/research/pubfiles/5228/better choosing experience.pdf](https://www0.gsb.columbia.edu/mygsb/faculty/research/pubfiles/5228/better%20choosing%20experience.pdf)
https://platzi.com/clases/scikitlearn-ml/

***
 Made by <a href='https://www.linkedin.com/in/novelo-luis/'> Luis Novelo </a>