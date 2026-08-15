import Modal from './Modal';
import FlipClock from '../flipclock/FlipClock';

interface FlipClockModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function FlipClockModal({ isOpen, onClose }: FlipClockModalProps) {
  return (
    <Modal isOpen={isOpen} onClose={onClose} fullScreen>
      <FlipClock />
    </Modal>
  );
}
