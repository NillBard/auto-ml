import { useEffect, useRef } from 'react';
import './StreamViewer.css';

interface StreamViewerProps {
  streamId: string;
}

export const StreamViewer: React.FC<StreamViewerProps> = ({ streamId }) => {
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Создаем новый EventSource при монтировании компонента
    eventSourceRef.current = new EventSource(`api/processing/streams/${streamId}/events`);

    // Обработчик сообщений
    eventSourceRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'frame') {
        const img = document.getElementById('stream-frame') as HTMLImageElement;
        if (img) {
          img.src = data.data.url;
        }
      }
    };

    // Обработчик ошибок
    eventSourceRef.current.onerror = (error) => {
      console.error('EventSource failed:', error);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };

    // Очистка при размонтировании компонента
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, [streamId]); // Пересоздаем EventSource при изменении streamId

  return (
    <div className="stream-viewer">
      <img
        id="stream-frame"
        alt="Stream frame"
        className="stream-frame"
      />
    </div>
  );
}; 