# Capstone project

## Prompt injection

El prompt injection es una vulnerabilidad en la que un atacante introduce instrucciones maliciosas en la entrada de un modelo de lenguaje (LLM), logrando que el modelo cambie su comportamiento y ejecute órdenes no previstas originalmente

### Técnicas

- Inyección directa: El atacante introduce comandos explícitos en la interfaz del modelo, como “Ignora todas las instrucciones anteriores” o adoptar roles persuasivos para que el modelo obedezca. Algunas subtécnicas son:
  1. Adición de reglas: Se agregan nuevas instrucciones que contradicen las existentes.</br>
  2. Negación de reglas: Se indica que ciertas restricciones ya no aplican.</br>
  3. Supresión de rechazos: Se fuerza al modelo a no rechazar ninguna petición.</br>
  4. Prompting de caso especial: Se convence al modelo de que la situación actual es una excepción legítima.</br>
- Inyección indirecta: Se ocultan comandos maliciosos dentro de contenido externo, como documentos, páginas web o correos electrónicos, que el modelo procesa sin saber que contienen instrucciones ocultas. Esto incluye técnicas de data poisoning y manipulación de pipelines RAG.
- Inyección persistente o almacenada: Los prompts maliciosos se guardan en bases de datos, historiales de chat o sistemas de conocimiento, activándose cuando el modelo los revisita.
- Técnicas evasivas y cognitivas: Incluyen hacking cognitivo, sidestepping, role-playing, escenarios hipotéticos, asignación de personalidad y deflexión de tareas. Estas buscan manipular el razonamiento del modelo o evadir restricciones sin ser detectadas.
