import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Capa LoRA (Low-Rank Adaptation) para transformaciones lineales."""
    
    def __init__(self, original_layer, rank=4, alpha=1.0):
        """
        Inicializa la capa LoRA.
        
        Args:
            original_layer: La capa linear original a envolver
            rank: El rango de la descomposición LoRA
            alpha: Parámetro de escalado para LoRA
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Matrices LoRA: A (down-projection) y B (up-projection)
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
        # Congelar parámetros originales para que no se entrenen
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Paso hacia adelante a través de la capa LoRA.
        
        El paso forward de LoRA debe:
        1. Calcular la salida de la capa original
        2. Calcular el camino LoRA: x -> A -> B -> escalar
            2a: Aplicar Capa A (down-projection)
            2b: Aplicar Capa B (up-projection)  
            2c: Multiplicar por la escala
        3. Añadir la salida LoRA a la salida original
        
        Args:
            x: Tensor de entrada de forma (batch, seq, in_features)
            
        Returns:
            Tensor de salida de forma (batch, seq, out_features)
        """
        # Calcular la salida de la capa original
        original_output = self.original_layer(x)

        # Calcular el camino LoRA: x -> A -> B con escalado
        # Para F.linear necesitamos transponer las matrices correctamente
        # lora_A: (rank, in_features) -> necesitamos (rank, in_features) como peso en F.linear
        lora_output = torch.matmul(x, self.lora_A.t())  # x @ A.T = (batch*seq, in_features) @ (in_features, rank)
        # lora_B: (out_features, rank) -> es correcto para F.linear
        lora_output = F.linear(lora_output, self.lora_B)  # (batch*seq, rank) con peso (out_features, rank)
        lora_output = lora_output * self.scaling  # Aplicar escalado

        # Añadir la salida LoRA a la salida original (conexión residual)
        return original_output + lora_output


def apply_lora(model, rank=4, alpha=1.0):
    """
    Aplica LoRA a las capas de proyección de atención en el modelo.
    
    Args:
        model: El modelo transformer a modificar
        rank: Parámetro de rango LoRA
        alpha: Parámetro alfa LoRA
        
    Returns:
        El modelo modificado con LoRA aplicado
    """
    # Aplicar LoRA recursivamente a todos los módulos
    modules_replaced = 0
    
    def apply_lora_recursive(parent_module, module_name=""):
        nonlocal modules_replaced
        
        for name, child_module in parent_module.named_children():
            full_name = f"{module_name}.{name}" if module_name else name
            
            # Verificar si este módulo debe ser reemplazado
            if isinstance(child_module, nn.Linear):
                # Solo aplicar LoRA a las proyecciones de atención específicas
                if any(proj in full_name for proj in ['compute_query', 'compute_key', 'compute_value', 'compute_output']):
                    lora_module = LoRALayer(child_module, rank=rank, alpha=alpha)
                    setattr(parent_module, name, lora_module)
                    modules_replaced += 1
                    print(f"LoRA aplicado a {full_name}")
            else:
                # Aplicar recursivamente a módulos hijo
                apply_lora_recursive(child_module, full_name)
    
    apply_lora_recursive(model)
    return model


def count_lora_parameters(model):
    """
    Cuenta los parámetros LoRA vs parámetros totales.
    
    Args:
        model: El modelo a analizar
        
    Returns:
        tuple: (lora_params, total_params, percentage)
    """
    lora_params = 0
    total_params = 0
    
    # Contar parámetros entrenable (requires_grad=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            # Si el nombre contiene 'lora_', es un parámetro LoRA
            if 'lora_' in name:
                lora_params += param_count
    
    percentage = (lora_params / total_params * 100) if total_params > 0 else 0
    return lora_params, total_params, percentage


def get_lora_optimizer_params(model):
    """
    Obtiene solo los parámetros LoRA para el optimizador.
    
    Args:
        model: El modelo del cual extraer los parámetros LoRA
        
    Returns:
        list: Lista de parámetros LoRA
    """
    lora_params = []
    # Solo incluir parámetros LoRA que requieren gradiente
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora_' in name:
            lora_params.append(param)
    return lora_params


def merge_lora_weights(model):
    """
    Fusiona los pesos LoRA de vuelta a las capas lineales originales.
    Esto crea un modelo estándar sin estructura LoRA.
    """
    def merge_lora_recursive(module):
        for name, child_module in module.named_children():
            if isinstance(child_module, LoRALayer):
                # Calcular el peso fusionado: W_original + (B @ A) * scaling
                # lora_A forma: (rank, in_features)
                # lora_B forma: (out_features, rank)
                # Necesitamos: lora_B @ lora_A para obtener (out_features, in_features)
                lora_weight = torch.mm(child_module.lora_B, child_module.lora_A) * child_module.scaling
                
                # Fusionar con el peso original
                merged_weight = child_module.original_layer.weight + lora_weight
                
                # Crear nueva capa lineal con pesos fusionados
                merged_layer = nn.Linear(
                    child_module.original_layer.in_features,
                    child_module.original_layer.out_features,
                    bias=child_module.original_layer.bias is not None
                )
                merged_layer.weight.data = merged_weight
                if child_module.original_layer.bias is not None:
                    merged_layer.bias.data = child_module.original_layer.bias.data
                
                # Reemplazar la capa LoRA con la capa fusionada
                setattr(module, name, merged_layer)
            else:
                merge_lora_recursive(child_module)
    
    merge_lora_recursive(model)
    return model