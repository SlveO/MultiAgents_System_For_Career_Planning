import { useRef } from 'react';
import { Paperclip, X } from 'lucide-react';

interface FileAttachmentProps {
  selectedFile: File | null;
  onFileChange: (file: File | null) => void;
}

export default function FileAttachment({ selectedFile, onFileChange }: FileAttachmentProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileChange(file);
    }
  };

  const handleRemove = () => {
    onFileChange(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="flex items-center gap-2">
      <input
        ref={inputRef}
        type="file"
        onChange={handleChange}
        className="hidden"
        id="file-upload"
      />
      <label
        htmlFor="file-upload"
        className="cursor-pointer p-2 rounded-lg hover:bg-gray-100 transition-colors text-gray-400 hover:text-primary-500"
        title="添加附件"
      >
        <Paperclip className="w-4 h-4" />
      </label>
      {selectedFile && (
        <div className="flex items-center gap-1.5 bg-primary-50 text-primary-700 text-xs px-2.5 py-1 rounded-full">
          <span className="max-w-[120px] truncate">{selectedFile.name}</span>
          <button
            onClick={handleRemove}
            className="hover:text-red-500 transition-colors"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}
    </div>
  );
}
