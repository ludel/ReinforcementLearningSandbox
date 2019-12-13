# Apprentissage par renfocement

- Environnement
    - stochastique: environnement dont les paramètres évoluent de manière aléatoire (prop d'effectuer une action != 1)
    - deterministe: réagit toujours de la même façon à un événement

- Diapos:
    - [Cours 1](https://docs.google.com/presentation/d/1YFeBtXoECruUklXCQTe8olgPqZTGmHQZ9vqrSOANhGk/edit#slide=id.g63da1a4385_0_0)
    - [Cours 2](https://docs.google.com/presentation/d/1ZqeG_44t_xtVw3QNd5bP4sE01DF6hlRkilfb316y9rY/edit#slide=id.g63da1a4385_0_0)
    - [Cours 3](https://docs.google.com/presentation/d/1JVoq354kPodTtU51rCA7MWXeXMR2mr0OdXsHpbMCvWM/edit#slide=id.g63da1a4385_0_0)

## Propriété de markov

> Les conséquences d’une action ne dépendent que de l’état actuel et non des états précédents

Un processus stochastique vérifie la propriété de Markov si et seulement si la distribution conditionnelle de 
probabilité des états futurs, étant donnés les états passés et l'état présent, ne dépend en fait que de l'état présent 
et non pas des états passés.

Si toutes les infos du problème sont données dans n'importe quel état.

## Equation de Bellman

Prendre le max pour les actions

## Utilité ou qualité (Item de la Q table)

L'objectif du système est de trouver les situations les plus utiles

L'utilité d'un état correspond au reward actuel plus les rewards qui peuvent suivre en fonction de leur proba et du discount factor. 
(point + ou - gros dans projet unity)

## Exploration

Initialement, le système ne connait pas les états ou se trouve les recompenses et les états finaux

## Formules Q learning

La taille de la Q-table corresponds le nombre d'actions X nombre d'états. 
Les deux algos reposent là dessus pour représenter l'apprentissage des différents actions


![qlearning](https://media.discordapp.net/attachments/644672581319786500/649003129731284993/unknown.png?width=400&height=32)

- alpha(α): learning rate
- r: reward
- gamma(γ): discount factor
- epsilon(ε): exploration rate
- a: action
- s: state
- pi: policy. Apprendre par l’expérience d’une stratégie comportementale. Dans nos cas c'est la methode choice_action()

### Politique greedy

Epsilon definit le type de politique
- Action non gluton: epsilon est grand, le random est important
- Action gluton: uilise forcement la meilleur action

## Methode de Mont-Carlo

La méthode Monte-Carlo désigne une famille de méthodes algorithmiques visant à calculer une valeur numérique approchée 
en utilisant des procédés aléatoires.

## Apprentissage en ligne ou hors ligne

Sarsa a tendance a aboutir à un résultat plus rapidement et que QLearning a tendance a explorer plus de possibilités

#### En ligne (Sarsa)
Sarsa est un algorithme "on-policy, il apprend directement des actions de l'agent. on prend l'action actuelle pour apprendre la QValue
    
Ces méthodes évaluent et améliorent la politique effectivement utilisée pour les décisions. Sarsa évalue la politique
 qu'il suit.
 
![sarsa](https://media.discordapp.net/attachments/644672581319786500/649003059350994962/unknown.png?width=400&height=30)

#### Hors ligne
La politique suivie est séparée de la politique évaluée. Qlearning évalue plusieurs politiques indépendamment de 
celle qu'il suit. 

On prend l'action avec la meilleur reward (plus grande QValue), pas nécessairement l'action actuelle

![qlearning](https://media.discordapp.net/attachments/644672581319786500/649003129731284993/unknown.png?width=400&height=32)

## Deep learning avec le renforcement

### Cerveau humain

100 milliard de neurones
Les réseaux de neurones sont une classe de modeles qui sont construits a l'aide de couche de neurones.
Deux principaux réseaux de neurones: convolutionnels et récurrents.

#### Neurones formel
- Les synapse sont représentés par des poids
- Corps cellulaire: fonction de transfert
- Axone: elements de sortie


### Fonction d'activation
Miner le fonctionnement d'un potentiel d'action d'un neurone. Il existe de neubreuses fonctions d'activation différentes.


### Features

#### Vision
Analyse d'image est la principal utilité de RN. Deux algo: classification de l'image puis localisation

#### Teste
Analyse d'entité et des relations dans un texte.

# Notes

[Livre](http://incompleteideas.net/book/bookdraft2017nov5.pdf)
