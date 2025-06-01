import logging
import asyncio
from kafka.errors import NoBrokersAvailable
from settings import settings
import pickle
from kafka import KafkaProducer, KafkaConsumer, KafkaClient, KafkaAdminClient
from kafka.admin import NewTopic
from schemas.pv_interface import Arguments, ActionCommand, OutputSwitchCommand, PsCommand, State

class ActionError(Exception):
    pass

class KafkaManager:
    def __init__(self):
        self.con = settings.KAFKA_URL
        logging.info(f"Initializing KafkaManager with URL: {self.con}")
        try:
            # Создаем топики, если их нет
            admin_client = KafkaAdminClient(bootstrap_servers=[self.con])
            topics = ['rtsp', 'rtsp-response']
            existing_topics = admin_client.list_topics()
            
            for topic in topics:
                if topic not in existing_topics:
                    logging.info(f"Creating topic: {topic}")
                    admin_client.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])
            
            self.producer = KafkaProducer(
                bootstrap_servers=[self.con],
                api_version=(0,11,5),
                request_timeout_ms=30000,
                retries=3,
                acks='all'
            )
            logging.info("Successfully created Kafka producer")
        except NoBrokersAvailable as err:
            logging.error(f"Failed to connect to Kafka: {err}")
            raise
        except Exception as e:
            logging.error(f"Error initializing Kafka: {str(e)}")
            raise

    async def action(self, rtsp_host, user, password, source_id, action):
        logging.info(f"Starting action: {action} for {rtsp_host}")
        arguments = Arguments(rtsp_host, user, password)
        message = ActionCommand(cmd_id='rest', source_id=source_id, action=action, args=arguments)
        
        # Создаем новый consumer для каждого запроса
        consumer = KafkaConsumer(
            "rtsp-response",
            bootstrap_servers=[self.con],
            reconnect_backoff_ms=1000,
            consumer_timeout_ms=30000,
            api_version=(0,11,5),
            auto_offset_reset='latest',
            group_id=None,
            enable_auto_commit=False
        )
        
        try:
            logging.info("Sending message to Kafka")
            # Отправляем сообщение и ждем подтверждения
            future = self.producer.send('rtsp', pickle.dumps(message, protocol=5))
            record_metadata = future.get(timeout=10)
            logging.info(f"Message sent successfully to partition {record_metadata.partition}, offset {record_metadata.offset}")
            
            # Ждем ответ в асинхронном режиме
            logging.info("Waiting for response from Kafka")
            for message in consumer:
                try:
                    logging.info(f"Received message from Kafka: {message}")
                    response = pickle.loads(message.value)
                    if not response.successful:
                        logging.error(f"Received unsuccessful response: {response.message}")
                        raise ActionError(response.message)
                    logging.info("Successfully processed response")
                    return response.successful
                except Exception as e:
                    logging.error(f"Error processing message: {str(e)}")
                    continue
                    
            logging.error("Timeout waiting for Kafka response")
            raise ActionError("No response received from Kafka")
                
        except Exception as e:
            logging.error(f"Error in Kafka action: {str(e)}")
            raise
        finally:
            # Всегда закрываем consumer
            try:
                consumer.close()
                logging.info("Consumer closed successfully")
            except Exception as e:
                logging.error(f"Error closing consumer: {str(e)}")

kafkaManager = KafkaManager()
