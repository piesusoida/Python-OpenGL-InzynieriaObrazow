import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]
EPSILON = 0.0001


def reflect(vector, normal_vector):
    """Returns reflected vector."""
    n_dot_l = np.dot(vector, normal_vector)
    return vector - normal_vector * (2 * n_dot_l)


def normalize(vector):
    """Return normalized vector (length 1)."""
    return vector / np.sqrt((vector**2).sum())

class Ray:
    """Ray class."""
    def __init__(self, starting_point, direction):
        """Ray consist of starting point and direction vector."""
        self.starting_point = starting_point
        self.direction = direction


class Light:
    """Light class."""
    def __init__(self, position):
        """The constructor.

        :param np.array position: The position of light
        :param np.array ambient: Ambient light RGB
        :param np.array diffuse: Diffuse light RGB
        :param np.array specular: Specular light RGB
        """
        self.position = position
        self.ambient=np.array([0, 0, 0])
        self.diffuse=np.array([0, 1, 1])
        self.specular=np.array([1, 1, 0])


class SceneObject:
    """The base class for every scene object."""
    def __init__(
        self,
        ambient=np.array([0, 0, 0]),
        diffuse=np.array([0.6, 0.7, 0.8]),
        specular=np.array([0.8, 0.8, 0.8]),
        shining=25
    ):
        """The constructor.

        :param np.array ambient: Ambient color RGB
        :param np.array diffuse: Diffuse color RGB
        :param np.array specular: Specular color RGB
        :param float shining: Shinging parameter (Phong model)
        """
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shining = shining

    def get_normal(self, cross_point):
        """Should return a normal vector in a given cross point.

        :param np.array cross_point
        """
        raise NotImplementedError

    def trace(self, ray):
        """Checks whenever ray intersect with an object.

        :param Ray ray: Ray to check intersection with.
        :return: tuple(cross_point, distance)
            cross_point is a point on a surface hit by the ray.
            distance is the distance from the starting point of the ray.
            If there is no intersection return (None, None).
        """
        raise NotImplementedError

    def get_color(self, cross_point, obs_vector, scene):
        """Returns a color of an object in a given point.

        :param np.array cross_point: a point on a surface
        :param np.array obs_vector: observation vector used in Phong model
        :param Scene scene: A scene object (for lights)
        """

        color = self.ambient * scene.ambient
        light = scene.light

        normal = self.get_normal(cross_point)
        light_vector = normalize(light.position - cross_point)
        n_dot_l = np.dot(light_vector, normal)
        reflection_vector = normalize(reflect(-1 * light_vector, normal))

        v_dot_r = np.dot(reflection_vector, -obs_vector)

        if v_dot_r < 0:
            v_dot_r = 0

        if n_dot_l > 0:
            color += (
                (self.diffuse * light.diffuse * n_dot_l) +
                (self.specular * light.specular * v_dot_r**self.shining) +
                (self.ambient * light.ambient)
            )

        return color


class Sphere(SceneObject):
    """An implementation of a sphere object."""
    def __init__(
        self,
        position,
        radius,
        ambient=np.array([0, 0, 0]),
        diffuse=np.array([0.6, 0.7, 0.8]),
        specular=np.array([0.8, 0.8, 0.8]),
        shining=25
    ):
        """The constructor.

        :param np.array position: position of a center of a sphere
        :param float radius: a radius of a sphere
        """
        super(Sphere, self).__init__(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            shining=shining
        )
        self.position = position
        self.radius = radius

    def get_normal(self, cross_point):
        return normalize(cross_point - self.position)

    def trace(self, ray):
        """Returns cross point and distance or None."""
        distance = ray.starting_point - self.position

        a = np.dot(ray.direction, ray.direction)
        b = 2 * np.dot(ray.direction, distance)
        c = np.dot(distance, distance) - self.radius**2
        d = b**2 - 4*a*c

        if d < 0:
            return (None, None)

        sqrt_d = d**(0.5)
        denominator = 1 / (2 * a)

        if d > 0:
            r1 = (-b - sqrt_d) * denominator
            r2 = (-b + sqrt_d) * denominator
            if r1 < EPSILON:
                if r2 < EPSILON:
                    return (None, None)
                r1 = r2
        else:
            r1 = -b * denominator
            if r1 < EPSILON:
                return (None, None)

        cross_point = ray.starting_point + r1 * ray.direction
        return cross_point, r1

class Triangle(SceneObject):
    def __init__(self, v0, v1, v2,
                 ambient=np.array([0.1, 0.5, 0.3]),
                 diffuse=np.array([0.6, 0.4, 0.2]),
                 specular=np.array([1.0, 0.8, 1.0]),
                 clarity = 0,
                 refraction = 0,                
                 shining=25):
        super().__init__(ambient, diffuse, specular, shining)
        self.v0 = np.array(v0)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.normal = normalize(np.cross(self.v1 - self.v0, self.v2 - self.v0))   # vektor normalny

    def get_normal(self, cross_point):
        return self.normal/np.sqrt((self.normal**2).sum()) # staly dla calego trojkata

    def trace(self, ray):
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        if abs(a) < EPSILON:
            return (None, None)  # figura plaska, dla rownoleglego promienia nie jest widoczny

        f = 1.0 / a
        s = ray.starting_point - self.v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return (None, None)

        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return (None, None)

        t = f * np.dot(edge2, q)
        if t > EPSILON:  # ray intersection
            cross_point = ray.starting_point + ray.direction * t
            return cross_point, t
        else:
            return (None, None)  # Line intersection but not a ray intersection

    
class MySphere(Sphere):
    def get_color(self, cross_point, obs_vector, scene):
        """Returns a color of an object in a given point.

        :param np.array cross_point: a point on a surface
        :param np.array obs_vector: observation vector used in Phong model
        :param Scene scene: A scene object (for lights)
        """
        color = self.ambient * scene.ambient
        light = scene.light

        normal = self.get_normal(cross_point)
        light_vector = normalize(light.position - cross_point)
        n_dot_l = np.dot(light_vector, normal)
        reflection_vector = normalize(reflect(-1 * light_vector, normal))

        interference = None
        for obj in scene.objects:
            first, second = obj.trace(Ray(cross_point, light_vector))
            if first is not None:
                interference = first
                break
        v_dot_r = np.dot(reflection_vector, -obs_vector)

        if v_dot_r < 0:
            v_dot_r = 0

        if n_dot_l > 0 and interference is None:
            color += (
                (self.diffuse * light.diffuse * n_dot_l) +
                (self.specular * light.specular * v_dot_r**self.shining) +
                (self.ambient * light.ambient)
            )

        return color
    
class Camera:
    """Implementation of a Camera object."""
    def __init__(
        self,
        position = np.array([0, 0, -3]),
        look_at = np.array([0, 0, 0]),
    ):
        """The constructor.

        :param np.array position: Position of the camera.
        :param np.array look_at: A point that the camera is looking at.
        """
        self.z_near = 1
        self.pixel_height = 500
        self.pixel_width = 700
        self.povy = 45
        look = normalize(look_at - position)
        self.up = normalize(np.cross(np.cross(look, np.array([0, 1, 0])), look))
        self.position = position
        self.look_at = look_at
        self.direction = normalize(look_at - position)
        aspect = self.pixel_width / self.pixel_height
        povy = self.povy * np.pi / 180
        self.world_height = 2 * np.tan(povy/2) * self.z_near
        self.world_width = aspect * self.world_height

        center = self.position + self.direction * self.z_near
        width_vector = normalize(np.cross(self.up, self.direction))
        self.translation_vector_x = width_vector * -(self.world_width / self.pixel_width)
        self.translation_vector_y = self.up * -(self.world_height / self.pixel_height)
        self.starting_point = center + width_vector * (self.world_width / 2) + (self.up * self.world_height / 2)

    def get_world_pixel(self, x, y):
        return self.starting_point + self.translation_vector_x * x + self.translation_vector_y * y


class Scene:
    """Container class for objects, light and camera."""
    def __init__(
        self,
        objects,
        light,
        camera
    ):
        self.objects = objects
        self.light = light
        self.camera = camera
        self.ambient = np.array([0.1, 0.1, 0.1])
        self.background = np.array([1, 1, 1])


class RayTracer:
    """RayTracer class."""
    def __init__(self, scene):
        """The constructor.

        :param Scene scene: scene to render
        """
        self.scene = scene

    def generate_image(self):
        """Generates image.

        :return np.array image: The computed image."
        """
        camera = self.scene.camera
        image = np.zeros((camera.pixel_height, camera.pixel_width, 3))
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                world_pixel = camera.get_world_pixel(x, y)
                direction = normalize(world_pixel - camera.position)
                image[y][x] = self._get_pixel_color(Ray(world_pixel, direction))
        return image

    def _get_pixel_color(self, ray):
        """Gets a single color based on a ray.

        :return np.array: A hit object color or a background.
        """

        obj, distance, cross_point = self._get_closest_object(ray)

        if not obj:
            return self.scene.background

        return obj.get_color(cross_point, ray.direction, self.scene)

    def _get_closest_object(self, ray):
        """Finds the closes object to the ray.

        :return tuple(scene_object, distance, cross_point)
            scene_object SceneObject: hit object or None (if no hit)
            distance float: Distance to the the scene_object
            cross_point np.array: the interection point.
        """
        closest = None
        min_distance = np.inf
        min_cross_point = None
        for obj in self.scene.objects:
            cross_point, distance = obj.trace(ray)
            if cross_point is not None and distance < min_distance:
                min_distance = distance
                closest = obj
                min_cross_point = cross_point

        return (closest, min_distance, min_cross_point)

class MyRayTracer(RayTracer):
    def _get_pixel_color(self, ray, depth=3):
        """Gets a single color based on a ray.

        :return np.array: A hit object color or a background.
        """

        obj, distance, cross_point = self._get_closest_object(ray)

        if not obj:
            return self.scene.background
        # po osiagnieciu limitu liczby odbic zwroc kolor obiektu
        if depth == 0:
            return obj.get_color(cross_point, ray.direction, self.scene)
        # w przeciwnym wypadku rekurencja, zwraca sume koloru obiektu i koloru odbicia
        
        # zmienna pomocnicza do obliczenia kierunku odbicia
        new_ray = Ray(cross_point, reflect(ray.direction, normalize(obj.get_normal(cross_point))))

        return ( 0.75 * obj.get_color(cross_point, ray.direction, self.scene) 
        + 0.25 * self._get_pixel_color( new_ray, depth=depth-1)
        )

class MySphere2(MySphere):
    def __init__(
        self,
        position,
        radius, 
        clarity=0,
        refraction=0,
        ambient=np.array([0, 0, 0]),
        diffuse=np.array([0.6, 0.7, 0.8]),
        specular=np.array([0.8, 0.8, 0.8]),
        shining=25
    ):
        super(MySphere2, self).__init__(
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
            shining=shining,
            position=position,
            radius=radius
        )
        self.clarity = clarity # poziom przezroczystosci
        self.refraction = refraction # wspolczynnik zalamania  
        
class MyRayTracer2(RayTracer):

    def refract(I, N, eta):
        
        cosi = -np.dot(N, I)
        sint2 = eta ** 2 * (1 - cosi ** 2)
        if sint2 > 1:
            return None  # całkowite wewnętrzne odbicie
        cost = np.sqrt(1 - sint2)
        return eta * I + (eta * cosi - cost) * N

    def _get_pixel_color(self, ray, depth=5):
        if depth == 0:
            return self.scene.background

        obj, distance, cross_point = self._get_closest_object(ray)

        if not obj:
            return self.scene.background

        normal = normalize(obj.get_normal(cross_point))
        local_color = obj.get_color(cross_point, ray.direction, self.scene)

        # odbicie
        reflected_dir = reflect(ray.direction, normal)
        reflected_origin = cross_point + EPSILON * normal
        reflected_ray = Ray(reflected_origin, reflected_dir)
        reflected_color = self._get_pixel_color(reflected_ray, depth - 1)

        if isinstance(obj, MySphere2) and obj.clarity > 0:
            # zakrzywienie
            refracted_dir = MyRayTracer2.refract(ray.direction, normal, obj.refraction)
            if refracted_dir is not None:
                refracted_dir = normalize(refracted_dir)
                refracted_origin = cross_point - EPSILON * normal
                refracted_ray = Ray(refracted_origin, refracted_dir)
                refracted_color = self._get_pixel_color(refracted_ray, depth - 1)

                # Weighted blend
                return (
                    (1 - obj.clarity) * local_color +
                    0.25 * reflected_color +
                    obj.clarity * 0.75 * refracted_color
                )

        # Opaque object
        return 0.85 * local_color + 0.15 * reflected_color


def test():
    scene = Scene(
        objects=[Sphere(position=np.array([0, 0, 0]), radius=1.5)],
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 5]))
    )

    rt = RayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    plt.imshow(image)
    plt.show()

def zadanie1():
    scene = Scene(
        objects=[
        Sphere(position=np.array([-2, 0, 0]), radius=2.25), 
        Sphere(position=np.array([3, -3, 0]), radius=0.7)
    ],
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 9]))
    )

    rt = RayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    plt.imshow(image)
    plt.show()

def zadanie2():
    scene = Scene(
        objects=[
        Sphere(position=np.array([-2, 0, 0]), radius=2.25, diffuse=np.array([1.0, 1.0, 0.0])), 
        Sphere(position=np.array([1, 1, 0]), radius=0.7, diffuse=np.array([0.0, 0.0, 1.0]))
    ],
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 9]))
    )

    rt = MyRayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    plt.imshow(image)
    plt.show()

def zadanie3():
    scene = Scene(
        objects=[
            MySphere(position=np.array([0, 0, 0]), radius=1.60), 
            MySphere(position=np.array([1, 1.2, 2]), radius=0.5, diffuse=np.array([0.5, 0.7, 0.3]))
        ],
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 8]))
    )


    rt = MyRayTracer(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    plt.imshow(image)
    plt.show()

def zadanie4():
    scene = Scene(
        objects=[
            MySphere2(position=np.array([0, 0, 0]), radius=1.60, clarity= 0.9, refraction= 0.1), 
            MySphere2(position=np.array([0, 0, -5]), radius=0.9, diffuse=np.array([0.5, 0.7, 0.3]))
        ],
        light=Light(position=np.array([3, 2, 5])),
        camera=Camera(position=np.array([0, 0, 8]))
    )


    rt = MyRayTracer2(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    plt.imshow(image)
    plt.show()

def zadanie5():
    scene = Scene(
        objects=[
            Triangle(
        v0=[-1, -1, 2],
        v1=[1, -1, 2],
        v2=[0, 1, 2],
        diffuse=np.array([1.0, 0.2, 0.2]),
        ambient=np.array([0.1, 0.1, 0.1]),
        specular=np.array([1.0, 1.0, 1.0]),
        shining=50
            ),
            MySphere2(position=np.array([4, 0, 0]), radius=1.60, clarity= 0.9, refraction= 0.1), 
            MySphere2(position=np.array([-4, 0, -5]), radius=0.9, diffuse=np.array([0.5, 0.7, 0.3]))
        ],
        
        light = Light(position=np.array([0, 2, 4])),
        camera = Camera(position=np.array([0, 0, 8]))
    )
    rt = MyRayTracer2(scene)
    image = np.clip(rt.generate_image(), 0, 1)
    plt.imshow(image)
    plt.show()

def menu():
    options = ["zadanie 1", "zadanie 2", "zadanie 3", "zadanie 4", "zadanie 5"]
    
    # Print the options
    print("Wybierz zadanie:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

    while True:
        try:
            # Ask the user to input a selection
            selection = int(input("Enter the number of your choice: "))
            
            # Store the input in a variable
            selected_option = selection
            
            # Check if the selection is valid
            if 1 <= selected_option <= 5:
                # Call the corresponding function based on the selection
                if selected_option == 1:
                    zadanie1()
                elif selected_option == 2:
                    zadanie2()
                elif selected_option == 3:
                    zadanie3()
                elif selected_option == 4:
                    zadanie4()
                elif selected_option == 5:
                    zadanie5()
                break  # Exit the loop if a valid option is selected
            else:
                print("Invalid selection. Please choose a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

menu()