class MobilePhone:
    def __init__(self, screen_type="Touch Screen", network_type="4G", dual_sim=False,
                 front_camera="8MP", rear_camera="12MP", ram="2GB", storage="16GB"):
        self.screen_type = screen_type
        self.network_type = network_type
        self.dual_sim = dual_sim
        self.front_camera = front_camera
        self.rear_camera = rear_camera
        self.ram = ram
        self.storage = storage

    def make_call(self, number):
        print(f"{self.__class__.__name__} is calling {number}.")

    def receive_call(self, number):
        print(f"{self.__class__.__name__} is receiving a call from {number}.")

    def take_a_picture(self, camera='rear'):
        if camera.lower() == 'front':
            print(f"Photo taken with {self.front_camera} front camera.")
        else:
            print(f"Photo taken with {self.rear_camera} rear camera.")

class Apple(MobilePhone):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.brand = "Apple"
        self.model = model

    def show_info(self):
        print(f"{self.brand} {self.model}: {self.ram} RAM, {self.storage} Storage, Dual SIM: {self.dual_sim}, "
              f"Front Camera: {self.front_camera}, Rear Camera: {self.rear_camera}, Network: {self.network_type}")

class Samsung(MobilePhone):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.brand = "Samsung"
        self.model = model

    def show_info(self):
        print(f"{self.brand} {self.model}: {self.ram} RAM, {self.storage} Storage, Dual SIM: {self.dual_sim}, "
              f"Front Camera: {self.front_camera}, Rear Camera: {self.rear_camera}, Network: {self.network_type}")

iphone_15 = Apple(
    model="iPhone 15 Pro",
    network_type="5G",
    dual_sim=True,
    front_camera="12MP",
    rear_camera="48MP",
    ram="8GB",
    storage="256GB"
)

iphone_13 = Apple(
    model="iPhone 13",
    network_type="5G",
    dual_sim=False,
    front_camera="12MP",
    rear_camera="12MP",
    ram="4GB",
    storage="128GB"
)

galaxy_s24 = Samsung(
    model="Galaxy S24 Ultra",
    network_type="5G",
    dual_sim=True,
    front_camera="12MP",
    rear_camera="200MP",
    ram="12GB",
    storage="512GB"
)

galaxy_m32 = Samsung(
    model="Galaxy M32",
    network_type="4G",
    dual_sim=True,
    front_camera="20MP",
    rear_camera="64MP",
    ram="6GB",
    storage="128GB"
)

iphone_13.show_info()
iphone_15.make_call("8260586748")
galaxy_s24.show_info()
galaxy_m32.take_a_picture('front')