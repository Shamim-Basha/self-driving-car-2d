import pygame
import math,random,sys,pickle,os
import neat
pygame.init()

WIDTH = 1200
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH,HEIGHT))
window = pygame.surface.Surface((screen.get_width(),screen.get_height()))
carImg = pygame.transform.rotate(pygame.image.load("./car.png"),90)

show_radar = False

class Car:
    def __init__(self,x,y) -> None:
        self.alive = True
        self.width = 20
        self.height = 40
        self.speed = 2
        self.img = pygame.transform.scale(carImg,(self.width,self.height))
        self.rotated_img = self.img
        self.rect = self.img.get_rect(topleft=(x,y))
        self.angle = 0
        self.radars = []
        self.radars_len = [0,0,0,0,0] #length of every radar
        self.reward = 0
    
    def draw(self):
        center = self.rect.center
        screen.blit(self.rotated_img,self.rect) #car
        if show_radar:
            for radar in self.radars:
                pygame.draw.line(screen,"#0000ff",(center),(radar)) #radar line
                pygame.draw.circle(screen,"#ff0000",(radar),5) #radar spot
        
    def rot_center(self):
        rot_image = pygame.transform.rotate(self.img, self.angle)
        rect = rot_image.get_rect(center = self.rect.center)
        return rot_image,rect
        
    def move(self):
        keys = pygame.key.get_pressed()
        dAngle = 2 #change of angle while turning
        angle = 360-self.angle
        
        self.rect.x += self.speed*math.sin(math.radians(angle))
        self.rect.y -= self.speed*math.cos(math.radians(angle))
        
        if keys[pygame.K_a]: #turn left
            self.angle += dAngle
        if keys[pygame.K_d]: #turn right
            self.angle -= dAngle
            if self.angle <0:
                self.angle += 360

        self.angle %= 360 
        self.rotated_img,self.rect = self.rot_center()
    
    def create_radar(self):
        self.radars = []
        self.radars_len = []
        center = self.rect.center
        angle = 360-self.angle
        
        def calculate_distance_from_center(x,y):
            return math.sqrt((center[0]-x)**2 + (center[1]-y)**2) 
        
        for theta in [angle,(90+angle),(270+angle),(45+angle),(315+angle)]:
            x,y = center
            dist = 0
            while (0<=x<WIDTH and 0<=y<HEIGHT) and window.get_at_mapped((math.floor(x),math.floor(y))):
                x += math.sin(math.radians(theta))
                y -= math.cos(math.radians(theta))
                dist = calculate_distance_from_center(x,y)
                if dist >=150:
                    break
            self.radars.append((x,y))
            self.radars_len.append(dist)
    
    def update_reward(self,x0,y0):
        x1, y1 = self.rect.center
        self.reward += ((x1-x0) + (y1-y0))/10000
        self.reward += self.speed/100
    
    def update(self):
        x0,y0 = self.rect.center
        self.move()
        self.update_reward(x0,y0)
        self.create_radar()
        self.is_crashed()
            
    def is_crashed(self):
        try:
            self.alive = window.get_at_mapped(self.rect.center)!=0
        except:
            self.alive = False
        if not self.alive:
            self.reward -= 10
    
    def get_data(self):
        return self.radars_len
    
    def get_reward(self):
        return self.reward
    
class Road:
    def draw(self):
        l,_,r = pygame.mouse.get_pressed()
        x,y = pygame.mouse.get_pos()
        if l:
            pygame.draw.circle(window,"#ffffff",(x,y),15)
        elif r:
            pygame.draw.circle(window,"#000000",(x,y),15)
        
        
# car = Car(50,350)
# road = Road()
# running = True
# clock = pygame.time.Clock()
# start = False
# Font = pygame.font.SysFont("arial",20)

current_generation = 0
start_pos = None

def run_simulation(genomes, config):
    clock = pygame.time.Clock()
    global start_pos,show_radar
                
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car(start_pos[0],start_pos[1]))
    
    screen.blit(window,(0,0))
    cars[0].draw()
    pygame.display.update()

    # Font Settings & Loading Map
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0
    game = True
    while game:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    show_radar = not show_radar
                if event.key == pygame.K_ESCAPE:
                    game = False

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10 # Left
            elif choice == 1:
                car.angle -= 10 # Right
            elif choice == 2:
                if(car.speed - 2 >= 12):
                    car.speed -= 2 # Slow Down
            else:
                car.speed += 2 # Speed Up
        
        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.alive:
                still_alive += 1
                car.update()
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 150: # Stop After About some Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(window,(0,0))
        for car in cars:
            if car.alive:
                car.draw()
        
        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, "#ff0000")
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, "#ff0000")
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)
        pygame.display.update()
        clock.tick(60) # 60 FPS

if __name__ == "__main__":
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    #setting up road
    road = Road()
    start = False
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    start = True
            l,_,r = pygame.mouse.get_pressed()
            if event.type == pygame.MOUSEMOTION:
                road.draw()
        screen.blit(window,(0,0))
        pygame.display.update()
                
    while not start_pos:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP:
                start_pos = pygame.mouse.get_pos()
                
    if os.path.exists("winner.pkl"):
        with open("winner.pkl","rb") as f:
            genome = pickle.load(f)
        genomes = [(1,genome)]
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            run_simulation(genomes,config)
    else:
        # Run Simulation For A Maximum of 1000 Generations
        winner = population.run(run_simulation, 20)
        with open("winner.pkl","wb") as f:
            pickle.dump(winner,f)
            f.close()
        
    
