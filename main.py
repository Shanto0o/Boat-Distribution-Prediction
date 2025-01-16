import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Circle

# Liste pour enregistrer les coordonnées des gros et petits bateaux
gros_bateaux_coords = []
petits_bateaux_coords = []

# Variable pour suivre la phase actuelle
phase = "gros"
model = None  # Le modèle GMM
moyenne_bateaux_par_gros = 0  # Moyenne des petits bateaux par gros bateau
hypothetical_coords = []  # Coordonnées des gros bateaux hypothétiques
rayon_influence = 400  # Rayon d'influence (en pixels)
image_path = "imgcoo/test2.png"  # Chemin de l'image

def train_gmm():
    """Entraîne un modèle GMM basé sur les coordonnées des petits bateaux relatifs aux gros."""
    global model, moyenne_bateaux_par_gros
    
    if len(gros_bateaux_coords) == 0 or len(petits_bateaux_coords) == 0:
        print("Erreur : Besoin de données pour entraîner le modèle.")
        return

    # Convertir en numpy arrays
    gros_array = np.array(gros_bateaux_coords)
    petits_array = np.array(petits_bateaux_coords)

    # Calculer les positions relatives des petits bateaux
    data_relative = []
    petits_autour_gros = []

    for gros in gros_array:
        distances = np.linalg.norm(petits_array - gros, axis=1)
        nb_petits = np.sum(distances < rayon_influence)
        petits_autour_gros.append(nb_petits)

        for petit in petits_array:
            data_relative.append(petit - gros)

    # Afficher les données relatives pour validation
    plt.figure()
    plt.scatter(*zip(*data_relative), alpha=0.5)
    plt.title("Données relatives : Petits bateaux par rapport aux gros")
    plt.xlabel("x relatif")
    plt.ylabel("y relatif")
    plt.show()

    # Calculer la densité moyenne
    moyenne_bateaux_par_gros = np.mean(petits_autour_gros)
    print(f"Moyenne des petits bateaux par gros bateau : {moyenne_bateaux_par_gros:.2f}")

    data_relative = np.array(data_relative)

    # Entraîner un GMM sur les données relatives
    model = GaussianMixture(n_components=3, random_state=0).fit(data_relative)
    print("Modèle GMM entraîné avec succès.")
    print("Poids des composantes :", model.weights_)
    print("Moyennes des composantes :", model.means_)

def predict_positions(hypothetical_coords):
    """Prédit les positions des petits bateaux à partir des positions des gros."""
    global model, moyenne_bateaux_par_gros, rayon_influence

    if model is None:
        print("Erreur : Le modèle n'est pas encore entraîné.")
        return []

    predictions = []

    # Pour chaque gros bateau hypothétique
    for gros_hypothetical in hypothetical_coords:
        # Générer un nombre moyen d'échantillons basé sur l'entraînement
        nb_points = int(np.round(moyenne_bateaux_par_gros))  # Moyenne observée

        # Obtenir des échantillons relatifs à partir du modèle
        samples, _ = model.sample(nb_points)

        # Restreindre les échantillons à une certaine distance
        filtered_samples = []
        for sample in samples:
            if np.linalg.norm(sample) <= rayon_influence:  # Restriction basée sur le rayon d'influence
                filtered_samples.append(gros_hypothetical + sample)

        predictions.extend(filtered_samples)

    return np.array(predictions)


def on_click(event):
    """Fonction appelée lorsqu'un clic est détecté sur l'image."""
    global phase
    
    if event.xdata is not None and event.ydata is not None:
        coords = (event.xdata, event.ydata)

        if phase == "gros":
            gros_bateaux_coords.append(coords)
            print(f"Gros bateau enregistré : {coords}")
            plt.plot(event.xdata, event.ydata, 'ro')  # Point rouge pour gros bateaux

            # Dessiner un cercle rouge autour du gros bateau
            circle = Circle(coords, rayon_influence/2, color='red', fill=False, linewidth=1)
            plt.gca().add_patch(circle)
        elif phase == "petits":
            petits_bateaux_coords.append(coords)
            print(f"Petit bateau enregistré : {coords}")
            plt.plot(event.xdata, event.ydata, 'bo')  # Point bleu pour petits bateaux
        elif phase == "hypothetical":
            hypothetical_coords.append(coords)
            print(f"Gros bateau hypothétique : {coords}")
            plt.plot(event.xdata, event.ydata, 'ro')  # Rouge pour les gros hypothétiques

        plt.draw()

def on_key(event):
    """Fonction appelée lorsqu'une touche est pressée."""
    global phase, hypothetical_coords

    if event.key == "enter":
        if phase == "gros":
            phase = "petits"
            print("Phase 2 : Cliquez sur les petits bateaux.")
        elif phase == "petits":
            train_gmm()  # Entraîne le modèle après avoir collecté les données
            phase = "hypothetical"
            hypothetical_coords = []
            print("Phase 3 : Cliquez pour ajouter des gros bateaux hypothétiques sur une page blanche.")
            plt.close()  # Fermer la fenêtre actuelle pour passer à la page blanche
            prediction_phase()  # Lancer la phase de prédiction
        elif phase == "hypothetical":
            # Prédire les positions des petits bateaux
            predicted_positions = predict_positions(hypothetical_coords)

            # Afficher les prédictions
            for pos in predicted_positions:
                plt.plot(pos[0], pos[1], 'go')  # Points verts pour les petits bateaux prédits

            plt.draw()
            print("Prédictions complétées.")

def prediction_phase():
    """Créer une page blanche pour la phase de prédiction."""
    global phase, hypothetical_coords
    
    phase = "hypothetical"
    hypothetical_coords = []

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.invert_yaxis()  # Inverser l'axe Y pour correspondre aux clics visuels

    ax.set_title("Phase 3 : Cliquez pour ajouter des gros bateaux hypothétiques.")

    # Connecter les événements
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)


    plt.show()

# Charger et afficher l'image
def main(image_path=None):
    """Programme principal pour gérer les clics et les étapes."""
    global phase

    # Afficher une image ou commencer sur une page blanche
    fig, ax = plt.subplots(figsize=(12, 8))
    if image_path:
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_title("Phase 1 et 2: Cliquez sur les gros bateaux puis les petits.")
    else:
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
        ax.invert_yaxis()  # Inverser l'axe Y pour correspondre aux clics visuels
        ax.set_title("Phase 1 et 2 : Cliquez sur les gros bateaux puis les petits.")

    # Connecter les événements
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)


    plt.show()

    # Résultats finaux
    print("Coordonnées des gros bateaux :", gros_bateaux_coords)
    print("Coordonnées des petits bateaux :", petits_bateaux_coords)


main(image_path)
