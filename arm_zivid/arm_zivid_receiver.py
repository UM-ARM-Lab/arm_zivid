import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import ros2_numpy
import cv2
import argparse


class ImageSubscriber(Node):
    def __init__(self, topic_name='/zivid/rgb'):
        super().__init__('zivid_image_viewer')

        # Create subscription to the specified topic
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f'Subscribed to topic: {topic_name}')

    def image_callback(self, msg):
        # Convert ROS2 Image message to NumPy array
        img_bgr = ros2_numpy.numpify(msg)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Show image using OpenCV
        cv2.imshow("Zivid RGB", img_rgb)
        cv2.waitKey(0)  # Wait for key press to close window
        cv2.destroyAllWindows()

        # Shutdown after displaying the first image
        rclpy.shutdown()


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='View images from ROS2 Image topic')
    parser.add_argument('--topic', '-t', type=str, default='/zivid/rgb',
                       help='Topic name to subscribe to (default: /zivid/rgb)')
    parsed_args = parser.parse_args()
    
    rclpy.init(args=args)
    node = ImageSubscriber(parsed_args.topic)
    rclpy.spin(node)


if __name__ == '__main__':
    main()
