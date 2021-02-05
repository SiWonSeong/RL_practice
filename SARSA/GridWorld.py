class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self, action):
        if action == 0:
            self.move_right()
        elif action == 1:
            self.move_left()
        elif action == 2:
            self.move_up()
        elif action == 3:
            self.move_down()

        reward = -1
        done = self.is_done()
        return (self.y, self.x), reward, done

    def move_right(self):
        if self.x == 6:
            pass
        elif self.x == 1 and self.y in [0, 1, 2]:
            pass
        elif self.x == 3 and self.y in [2, 3, 4]:
            pass
        else:
            self.x += 1

    def move_left(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y in [0, 1, 2]:
            pass
        elif self.x == 5 and self.y in [2, 3, 4]:
            pass
        else:
            self.x -= 1

    def move_up(self):
        if self.y == 0:
            pass
        elif self.y == 3 and self.x == 2:
            pass
        else:
            self.y -= 1

    def move_down(self):
        if self.y == 4:
            pass
        elif self.y == 1 and self.x == 4:
            pass
        else:
            self.y += 1

    def is_done(self):
        if self.y == 4 and self.x == 6:  # 목표 지점인 (4, 6)에 도달하면 끝난다
            return True
        else:
            return False

    def get_state(self):  # State: y, x 좌표
        return self.y, self.x

    def reset(self):
        self.x = 0
        self.y = 2
        return self.get_state()
