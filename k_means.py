import pandas as pd

from sklearn.cluster import KMeans

if __name__ == '__main__':

    dataset = pd.read_csv('./in/candy.csv') #Ubicación de archivo fuente
    x = dataset.drop(['competitorname'], axis=1 )

    
    kmeans = KMeans(n_clusters=4, random_state=1).fit(x) #Usar n_clusters para el númer de grupos deseados, random_state se define con cualquier número para asegurar los mismos resultados
    print('='*64) #Esta línea es solo para estética de impresión en terminal
    print('Total de grupos:', len(kmeans.cluster_centers_))
    print('='*64)
    dataset['grupo'] = kmeans.predict(x) #Nombre de la nueva columna donde pondremos nuestra predicción de algoritmo
    dataset.to_csv('./out/groups.csv', index=False, header=True) #Creación de archivo en carpeta destino

    grouped_df = dataset.groupby(['grupo']) 

    for key, group in grouped_df:  #Muestra cada grupo
        print(grouped_df.get_group(key), "\n\n", '='*180)  

    print('El Archivo "groups.csv" ha sido creado con éxito en la carpeta "out".')

