# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:28:04 2021

@author: Administrador (Andrés Felipe Escallón Portilla)
"""
#importing modules
    
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import math
#import random
import json

#opening the eqpt_config_96chan.json file:
f = open('eqpt_config_96chan.json',) 
   
data = json.load(f) #data is a dictionary

max_length = data['Span'][0]['max_length'] # an EDFA (in-line amplifier) has to be placed every max_length km
print('max_length = ',max_length)#set to 80km

pd.options.display.max_columns = None #show all columns in a dataframe

#input files:
    #C:\Users\Administrador\my_python_scripts\andres-graphs-DES-repo-GitHub\modularized_program\input_files2
path_in ="./input_files2/"
file_in = path_in+"markov_chain_based_on_example_italian_net.xls"#everthing being read from a unique file
#file1 = path_in+"network_nodelist_italian.csv"# nodes are inside file_in
#file2 = path_in+"network_edgelist_italian.csv"# edges are inside file_in
file3 = path_in+"markov_chain_based_on_network_nodepositionlist_italian2.csv" # to be able to draw the graph
file4 = path_in+"markov_chain_based_on_network_weighted_graph_distance_matrix.csv"# to have an idea of the number of trajectories of length t (t is an integer>=2) going from source to target

#output files:
path_out ="./output_files/"
file_out = path_out+"markov_chain_based_on_example_italian_net_results.xlsx"#everthing being written to a unique file 

######################   BEGINNING OF THE DATA_TOPOLOGY SECTION ###############################################

#READING EXCEL FILES (~GNPy):
#https://www.analyticslane.com/2018/07/30/guardar-y-leer-archivos-excel-en-python/


df_network_nodelist = pd.read_excel(file_in, sheet_name='Nodes', 
                           skiprows = [1,2,3,4],# skipping the first 4 rows after the header
#the original header will be replaced by these columns in the 5th row to sart reading the file from the 6th row of the original (which is now the first row or position 0 of the new modified file)
                           names=['City','State','Country','Region','Latitude','Longitude','Type','Booster_restriction','Preamp_restriction'])
print('\ndf_network_nodelist =\n',df_network_nodelist)

df_network_edgelist = pd.read_excel(file_in, sheet_name='Links', 
                           skiprows = [1,2,3,4],# skipping the first 4 rows after the header
#the original header will be replaced by these columns in the 5th row to sart reading the file from the 6th row of the original (which is now the first row or position 0 of the new modified file)
                           names=['Node A','Node Z','Distance (km)','Fiber type','lineic att','Con_in','Con_out','Cable id','PMD','Distance (km)','Fiber type','lineic att','Con_in','Con_out','PMD','Cable id'])
#as columns in the above names are repeated in the original file (Links are bidirectional), then pandas assigns .1 at the end of every repeated column to differentiate them
print('\ndf_network_edgelist =\n',df_network_edgelist)


df_network_nodepositionlist = pd.read_csv(file3)
print('\ndf_network_nodepositionlist =\n',df_network_nodepositionlist)

df_network_weighted_graph_distance = pd.read_csv(file4)
print('\ndf_network_weighted_graph_distance =\n',df_network_weighted_graph_distance)


df_component_rates = pd.read_excel(file_in, sheet_name='component_rates',
                                 skiprows = [1,2,3,4],# skipping the first 4 rows after the header
#the original header will be replaced by these columns in the 5th row to sart reading the file from the 6th row of the original (which is now the first row or position 0 of the new modified file)
                                 names=['Component', 'λ (FIT)', 'µ (FIT)'])

print("\ndf_component_rates =\n",df_component_rates)

df_Service = pd.read_excel(file_in, sheet_name='Service', 
                           skiprows = [1,2,3,4],# skipping the first 4 rows after the header
                           names=['route id', 'Source', 'Destination','TRX type', 'Mode','System: spacing','System: input power (dBm)','System: nb of channels','routing: disjoint from','routing: path','routing: is loose?','path bandwidth'])
#the original header will be replaced by these columns in the 5th row to sart reading the file from the 6th row of the original (which is now the first row or position 0 of the new modified file)
print("\ndf_Service =\n",df_Service)


FIT=10**9

df_component_rates_2 = df_component_rates.copy()
df_component_rates_2['MTTF (h)'] = FIT/df_component_rates_2['λ (FIT)']
df_component_rates_2['MTTR (h)'] = FIT/df_component_rates_2['µ (FIT)']
df_component_rates_2['Availability'] = df_component_rates_2['MTTF (h)'] / (df_component_rates_2['MTTF (h)'] + df_component_rates_2['MTTR (h)'])

print("\ndf_component_rates_2 =\n",df_component_rates_2)

#saving dataframes in different sheets of a same .xlsx file:
#REFERENCES:
#    https://www.analyticslane.com/2018/07/30/guardar-y-leer-archivos-excel-en-python/
#    https://www.analyticslane.com/2020/07/06/guardar-diferentes-hojas-excel-con-python/
#    https://www.it-swarm-es.com/es/python/pandas-to-csv-primera-columna-extra-eliminar-como/1048866204/

writer = pd.ExcelWriter(file_out)
df_component_rates_2.to_excel(writer, sheet_name='component_rates_2', index = False)#index = False (does not show the index column)


#ADJANCENCY MATRIX:
df_network_graph_adjacency = df_network_weighted_graph_distance.replace(np.nan,0)
#print("\ndf_network_graph_adjacency:\n",df_network_graph_adjacency)
#print(df_network_graph_adjacency.columns)
#print(len(df_network_graph_adjacency))
#print(type(df_network_graph_adjacency['Node'][0]))

df_network_graph_adjacency_def = df_network_graph_adjacency[df_network_graph_adjacency.columns[1::]].copy()
#print("\ndf_network_graph_adjacency COPIED:\n",df_network_graph_adjacency_def)

for i in df_network_graph_adjacency_def.columns:
    for j in range(len(df_network_graph_adjacency_def)):    
        if (df_network_graph_adjacency_def[i][j]!=0):        
            df_network_graph_adjacency_def[i][j]=1

#print("\ndf_network_graph_adjacency DEF:\n",df_network_graph_adjacency_def)

matrix_network_graph_adjacency_def = df_network_graph_adjacency_def.to_numpy()
#print("\nmatrix__network_graph_adjacency DEF:\n",matrix_network_graph_adjacency_def)

#[A(D)]^t gives the number of trajectories of lenght t (t is an integer>=2) that go from source to target (Ui-Uj):

#Initializing...
matrix_network_graph_adjacency_def_t = matrix_network_graph_adjacency_def.copy()

#Looping t from 2 to 5:

for t in range(2,6,1):
    message = f'[A(D)]^{t} gives the number of trajectories of lenght {t} that go from source to target (Ui-Uj)'
    matrix_network_graph_adjacency_def_t = np.matmul(matrix_network_graph_adjacency_def, matrix_network_graph_adjacency_def_t)    
    #print(message+f'\nmatrix_network_graph_adjacency_def^{t}:\n{matrix_network_graph_adjacency_def_t}')

#IT WORKED!!!
#Manually comproving:

#[A(D)]^t gives the number of trajectories of lenght t that go from source to target (Ui-Uj)
matrix_network_graph_adjacency_def_t2 = np.matmul(matrix_network_graph_adjacency_def, matrix_network_graph_adjacency_def)
#print('\nmatrix_network_graph_adjacency_def^2:\n',matrix_network_graph_adjacency_def_t2)

matrix_network_graph_adjacency_def_t3 = np.matmul(matrix_network_graph_adjacency_def, matrix_network_graph_adjacency_def_t2)
#print('\nmatrix_network_graph_adjacency_def^3:\n',matrix_network_graph_adjacency_def_t3)

matrix_network_graph_adjacency_def_t4 = np.matmul(matrix_network_graph_adjacency_def, matrix_network_graph_adjacency_def_t3)
#print('\nmatrix_network_graph_adjacency_def^4:\n',matrix_network_graph_adjacency_def_t4)

matrix_network_graph_adjacency_def_t5 = np.matmul(matrix_network_graph_adjacency_def, matrix_network_graph_adjacency_def_t4)
#print(message+'\nmatrix_network_graph_adjacency_def^5:\n',matrix_network_graph_adjacency_def_t5)

      
#ITALIAN NETWORK TOPLOGY TRHOUGH A GRAPH
G = nx.Graph() # Creating an empty undirected graph
#print(type(G))  # Verification of G data type

nodelist = list(df_network_nodelist["City"])
#print(nodelist)

#edgelist_source = list(df_network_edgelist["Node A"])
#print(edgelist_source)
#edgelist_target = list(df_network_edgelist["Node Z"])
#print(edgelist_target)

#edge_tuples_list = [(x,y) for (edgelist_source,edgelist_target) in range(len(edgelist_source))]
#edge_tuples_list = [(edgelist_source[i],edgelist_target[i]) for i in range(len(edgelist_source))]

edge_tuples_list = [(df_network_edgelist["Node A"][i],df_network_edgelist["Node Z"][i]) for i in range(len(df_network_edgelist))]
#print('\nedge_tuples_list =\n ', edge_tuples_list)

# Adicion de una coleccion de nodos
#G.add_nodes_from(['N1', 'N2', 'N3', 'N4'])
G.add_nodes_from(nodelist)
#print('\nG.nodes() = \n',G.nodes()) #It prints the node list

# Los nodos deben existir
# Adicion desde una lista de ejes

#G.add_edges_from([("N1", "N2"), ("N2", "N3"),("N2", "N4")])
G.add_edges_from(edge_tuples_list)
#print('\nG.edges() = \n',G.edges()) ##It prints the edge list

#nx.draw(G, with_labels=True) # grafico por defecto

# Dibujando un grafo con una pantilla de diseño
#nx.draw_shell(G,with_labels=True)

# Dibujando los nodos
#nx.draw_networkx_nodes(G,pos=nx.spring_layout(G))
#nx.draw_circular(G,with_labels=True)
#nx.draw_kamada_kawai(G,with_labels=True)
#nx.draw_planar(G,with_labels=True)
#nx.draw_random(G,with_labels=True)
#nx.draw_spectral(G,with_labels=True)
#nx.draw_networkx_edges(G,{"N1":1,"N2":2,"N3":3,"N4":4})
#nx.draw_networkx_edges(G,{["N1","N2","N3","N4"]:[]})
#nx.draw_networkx_edges(G,pos=layout)

#explicitly set positions

#pos = {"N1": (40, 60),
#       "N2": (0, 90),
#       "N3": (100, 80),
#       "N4": (0, 0),
#       "N5": (100, 0)
#       }

node_position_dict = dict([(df_network_nodepositionlist["Node"][i], (df_network_nodepositionlist["posX (km)"][i],df_network_nodepositionlist["posY (km)"][i])) for i in range(len(df_network_nodepositionlist))])
#print(node_position_dict)

#lista_nodos = ["N1","N2","N3","N4"]
#lista_pesos_aristas = ["30","50","50"]

# Dibujar nodo
#nx.draw_networkx_nodes(G, pos, node_size=300, nodelist=lista_nodos, node_color='r', label=lista_nodos)
# Dibujar líneas
#nx.draw_networkx_edges(G, pos, alpha=0.25, edge_color='b', width=3)
#plt.axis('off')

# asignar atributos a nodos y aristas
#G.nodes["N1"]["type"] = "OADM"
#G.nodes["N2"]["type"] = "ROADM"
#G.nodes["N3"]["type"] = "OADM"
#G.nodes["N4"]["type"] = "OADM"

#print('\nG.nodes(data=True) BEFORE =',G.nodes(data=True))
#print('\nG.edges(data=True) BEFORE =',G.edges(data=True))
#print('\nG["N1"] BEFORE =',G["N1"])
    
for i in range(len(df_network_nodelist)):
    G.nodes[df_network_nodelist["City"][i]]["type"] = df_network_nodelist["Type"][i]
    
#UNNECESSARY BECAUSE THE WEIGHT WERE ASSIGNED IN THE FOLLOWING LINES WITH: G.add_weighted_edges_from(array_export)
#for i in range(len(df_network_nodelist)):
#    G.edges[df_network_edgelist["Distance (km)"][i]]["weight"] = df_network_edgelist["Distance (km)"][i]

#G.nodes["N2"] = "ROADM"
#G.nodes["N3"] = "OADM"
#G.nodes["N4"] = "OADM"
#G.edges[("N1","N2")] = 30
#G.edges[("N2","N3")] = 50
#G.edges[("N2","N4")] = 50

#print('\nG.nodes(data=True) =',G.nodes(data=True))
#print('\nG.edges(data=True) =',G.edges(data=True))
#print('\nG["N1"] =',G["N1"])
#nx.draw(G, with_labels=True) # grafico por defecto

#source = ["N1","N2","N2"]
source = list(df_network_edgelist["Node A"])
#print(source)
#target = ["N2","N3","N4"]
target = list(df_network_edgelist["Node Z"])
#print(target)
#weight = [30,50,50]
#weight = ["30km","50km","50km"]
weight = list(df_network_edgelist["Distance (km)"])
#print(weight)

array_export = [(source[i], target[i], weight[i]) for i in range(len(source))] #range(source.size)]

G.add_weighted_edges_from(array_export)

list_edges = list(G.edges())

nx.draw(G, pos=node_position_dict, with_labels=True, edgelist=list_edges, node_size=150)

edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])

nx.draw_networkx_edge_labels(G, pos=node_position_dict, edgelist=list_edges, edge_labels=edge_labels)
plt.axis('off')
plt.show()
#print(G.nodes(data=True))
#print(G.edges(data=True))
#print(G["N1"])

#source = input("source: ")
#target = input("target: ")

#https://www.delftstack.com/es/howto/python/python-randomly-select-from-list/
#source = random.choice(source)
#target = random.choice(target)
#print('random source= ',source)
#print('random target= ',target)


print('\nG.nodes(data=True) = \n',G.nodes(data=True)) #It prints the node list
print('\nG.edges(data=True) = \n',G.edges(data=True)) ##It prints the edge list
print('\nG.edges[("N1", "N2") = ',G.edges[('N1', 'N2')]) #G.edges[('N1', 'N2')] = {'weight': 140}      
print('\nG[("N1", "N2")]["weight"]  = ',G.edges[('N1', 'N2')]['weight']) #G.edges[('N1', 'N2')]['weight'] = 140      
print('\nG.nodes["N1"] =',G.nodes["N1"])#G.nodes["N1"]={'type': 'ROADM'}
print('\nG.nodes["N1"]["type"] =',G.nodes["N1"]["type"])#G.nodes["N1"]["type"]=ROADM
print('\nG["N1"] =',G["N1"])#G["N1"] = {'N2': {'weight': 140}, 'N6': {'weight': 210}, 'N3': {'weight': 110}}
print('\nG["N1"]["N2"]["weight"] =',G["N1"]["N2"]['weight'])
print('\nsource = ',source)
print('\ndestination = ',target)

print('\ndf_network_edgelist =',df_network_edgelist)

print("\nThis is the end of the data_topology section!")

######################   END OF THE DATA_TOPOLOGY SECTION FUNCTIONS   ###############################################


#DONE: make a for cicle to get a list of source and a list of destination
#DONE: perform the cicle for each availability conecction saving in a different xls
#make part of the following code a fuction to perform availability_connection_calculus(source,target)
#loop the function over index:

#source = df_Service['Source'][5]
#target = df_Service['Destination'][5]


'''
######################   BEGINNING OF THE AVAILABILITY SECTION ###############################################

def return_distance_from_links(G,source,target):
    shortest_path_list = nx.dijkstra_path(G,source,target)#assumes weight=1
    #shortest_path_list = nx.dijkstra_path(G,source,target,weight=G.edges[source,target]['weight'])
    #shortest_path_list = nx.dijkstra_path(G,source,target,weight=peso[(u,v)])
    print("\nshortest_path_list = ",shortest_path_list,", lenght = ",len(shortest_path_list))#-1)    
    #Dijstra with weigts different to 1: now trying...
    #REFERENCES:
    #    https://www.analyticslane.com/2019/06/21/seleccionar-filas-y-columnas-en-pandas-con-iloc-y-loc/
    #    https://www.delftstack.com/es/howto/python-pandas/how-to-get-index-of-all-rows-whose-particular-column-satisfies-given-condition-in-pandas/
    #    https://www.delftstack.com/es/howto/python-pandas/pandas-get-index-of-row/
    #    https://www.analyticslane.com/2018/07/30/guardar-y-leer-archivos-excel-en-python/    
    distance_list = []
    #source=Node A, target=Node Z
    for i in range(len(shortest_path_list)-1):        
        #index = df_network_edgelist.index[(df_network_edgelist["Node A"]==shortest_path_list[i])&(df_network_edgelist["Node Z"]==shortest_path_list[i+1])].tolist()
        index = df_network_edgelist.index[((df_network_edgelist["Node A"]==shortest_path_list[i])&(df_network_edgelist["Node Z"]==shortest_path_list[i+1]))|((df_network_edgelist["Node A"]==shortest_path_list[i+1])&(df_network_edgelist["Node Z"]==shortest_path_list[i]))].tolist()#bidirectional
        #print('index =',index)
        distance_list.append(df_network_edgelist["Distance (km)"][index[0]])
    return distance_list

def availability_connection_calculus(G,source,target,max_length):
    #pesos = G.edges()#[source,target][weight]
    #pesos = G.edges[source,target]['weight']
    #pesos = edge_labels[(source,target)]
    #print("pesos :",pesos)#,"len = ", len(pesos))
    #u = source
    #v = target
    #peso = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
    #print("\npeso: ",peso)    
    distance_list = return_distance_from_links(G,source,target)
    print("\ndistance_list =", distance_list)
    print(f"\nTotal distance covered = {sum(distance_list)} km")
    
    hop_list = [i+1 for i in range(len(distance_list))]
    print("\nhop_list =", hop_list)
    
    max_length = max_length #80km 
    
    num_span_list = [math.ceil(distance_list[j]/max_length) for j in range(len(distance_list))]
    print("\nspan_list =", num_span_list)
    
    num_amplifier_list = [2+(num_span_list[j]-1) for j in range(len(num_span_list))]
    print("\nnum_amplifier_list =", num_amplifier_list)
    
    #A-services-italian-net-GNPy:
    #https://www.youtube.com/watch?v=DuD6wODeelE
    dict_of_lists = {'Hop':hop_list,'Distance (km)':distance_list,'Num_spans':num_span_list,'Num_amplifiers':num_amplifier_list}    
    df = pd.DataFrame(data=dict_of_lists)
    #list_of_lists = list(zip(hop_list,distance_list, num_span_list,num_amplifier_list))#zip: list of tuples
    ##list_of_lists = [hop_list,distance_list, num_span_list,num_amplifier_list]
    #print(list_of_lists)
    #df = pd.DataFrame(list_of_lists)
    #print('df =')
    #print(df)
    #print()
    #print(df.columns)
    ##df = pd.DataFrame(list_of_lists,
    ##                  columns=['Hop','Distance (km)', 'Num_spans', 'Num_amplifiers'])
    ##print('df =')
    ##print(df)
    ##print()
    ##print(df.columns)
    ##df2 = df.rename(columns={0:'Hop',1:'Distance (km)',2:'Num_spans',3:'Num_amplifiers'})
    #df.rename(columns={0:'Hop',1:'Distance (km)',2:'Num_spans',3:'Num_amplifiers'},inplace=True)
    print('\ndf =\n',df)
    
    df['A_amplifier'] = df_component_rates_2['Availability'][1]**df['Num_amplifiers']
    df['λ (FIT)'] = df_component_rates_2['λ (FIT)'][2]*df['Distance (km)']
    df['µ (FIT)'] = df_component_rates_2['µ (FIT)'][2]
    df['MTTF (h)'] = FIT/df['λ (FIT)']
    df['MTTR (h)'] = FIT/df['µ (FIT)']
    df['A_fiber'] = df['MTTF (h)'] / (df['MTTF (h)'] + df['MTTR (h)'])
    print("df=\n",df)
    
    #https://www.it-swarm-es.com/es/python/pandas-to-csv-primera-columna-extra-eliminar-como/1048866204/
    #df.to_excel(file5_3, sheet_name='availability_services', index = False)#index = False (does not show the index column)
    #df_component_rates_2[df_component_rates_2.columns[1::]].to_excel(file5_2_2, sheet_name='component_rates_2')
    
    #Ac = (A_nodes)*(A_amplifiers)*(A_fibers)
    A_nodes = df_component_rates_2['Availability'][0]**(len(df['Hop'])+1)
    print('\nA_nodes =',round(A_nodes,6))
    
    A_amplifiers = df['A_amplifier'].product()
    print('\nA_amplifiers =',round(A_amplifiers,6))
    
    A_fibers = df['A_fiber'].product()
    print('\nA_fibers =',round(A_fibers,6))
    
    Availability_connection = (A_nodes)*(A_amplifiers)*(A_fibers)
    
    print('\nAvailability_connection = ',round(Availability_connection,6))
    
    return df, Availability_connection, 
    

source = [] 
target = []
Availability_connection = []



for i in range(len(df_Service)):
    source.append(df_Service['Source'][i])
    target.append(df_Service['Destination'][i])
    print('\nsource = ',source[i])
    print('\ndestination = ',target[i])
    (df, availability_connection) = availability_connection_calculus(G,source[i],target[i],max_length)
    df.to_excel(writer, sheet_name="A_service_"+str(i), index=False)#it is dinamically changed over a loop (several sheest in a unique file)
    Availability_connection.append(availability_connection)
    

print('\nsource_list = ',source)
print('\ndestination_list = ',target)
print('\nAvailability_connection_list =\n ',Availability_connection)

df_Service['Availability_connection'] = pd.Series(Availability_connection)
print('\ndf_Service=\n',df_Service)


df_Service.to_excel(writer, sheet_name='df_Service', index = False)#index = False (does not show the index column)

writer.save()
writer.close()

print("This is the of the availability section!")
######################   END OF OF THE AVAILABILITY SECTION  ###############################################

'''
print("This is the end ot the program because everything is working properly!")
