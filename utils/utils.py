from dataclasses import dataclass

@dataclass
class BBoxModel:
    """
    Define o formato de bbox que será utilizado nos resultados dos modelos.

    Attributes:
        x: coordenada x do ponto superior esquerdo do bbox.
        y: coordenada y do ponto superior esquerdo do bbox.
        w: largura do bbox.
        h: altura do bbox.
    """
    x: int
    y: int
    w: int
    h: int

    def __iter__(self):
        """
        Itera sobre os atributos do bbox.
        """
        yield self.x
        yield self.y
        yield self.w
        yield self.h


    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """
        Retorna as coordenadas em inteiro (x1, y1, x2, y2) para uso com crop
        """
        x1 = self.x
        y1 = self.y
        x2 = self.x + self.w
        y2 = self.y + self.h

        return int(x1), int(y1), int(x2), int(y2)

    @classmethod
    def xy_to_xyhw(cls, x1, y1, x2, y2, img_shape):
        """
        Converte um resultado no formato x1, y1, x2, y2 para o formato x, y, w, h.


        Args:
            x1 : coordenada x do ponto superior esquerdo do bbox.
            y1 : coordenada y do ponto superior esquerdo do bbox.
            x2 : coordenada x do ponto inferior direito do bbox.
            y2 : coordenada y do ponto inferior direito do bbox.
            img_shape : shape da imagem de entrada.

        Returns:
            BBoxModel : bbox no formato x, y, w, h.
        """
        h, w = img_shape[:2]
        real_y1 = y1 * h
        real_x1 = x1 * w
        real_y2 = y2 * h
        real_x2 = x2 * w

        real_w = real_x2 - real_x1
        real_h = real_y2 - real_y1

        return BBoxModel(y=(real_y1), x=(real_x1), h=real_h, w=real_w)


@dataclass
class DetectedObject:
    """
    Define o formato de retorno de um unico resultado de uma inferencia.

    Attributes:
        bbox (BBoxModel) : BBox do objeto identificado
        label (str) : Label do objeto detectado (carro, moto, etc.)
        confidence (float) : Confianca da deteccao do objeto

    """
    bbox: BBoxModel
    label: str
    confidence: float
