from config import Config
from agent import Agent
from data_util import sample_data


def main():
    agent = Agent(Config)
    agent.make_tsne_pic_for_folder()
    # img = sample_data(1)
    # bounding_box, mask = agent.get_bounding_box(img)
    # print(bounding_box)


if __name__ == '__main__':
    main()
