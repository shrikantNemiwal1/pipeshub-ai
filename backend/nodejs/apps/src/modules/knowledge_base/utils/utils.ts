import FormData from 'form-data';
import { AuthenticatedUserRequest } from '../../../libs/middlewares/types';
import { Logger } from '../../../libs/services/logger.service';
import { FileBufferInfo } from '../../../libs/middlewares/file_processor/fp.interface';
import axios from 'axios';
import { KeyValueStoreService } from '../../../libs/services/keyValueStore.service';
import { endpoint } from '../../storage/constants/constants';
import { HTTP_STATUS } from '../../../libs/enums/http-status.enum';
import { DefaultStorageConfig } from '../../tokens_manager/services/cm.service';

const logger = Logger.getInstance({
  service: 'knowledge_base.utils',
});

const axiosInstance = axios.create({
  maxRedirects: 0,
});

export interface StorageResponseMetadata {
  documentId: string;
  documentName: string;
}

export const saveFileToStorageAndGetDocumentId = async (
  req: AuthenticatedUserRequest,
  file: FileBufferInfo,
  documentName: string,
  isVersionedFile: boolean,
  keyValueStoreService: KeyValueStoreService,
  defaultConfig: DefaultStorageConfig,
): Promise<StorageResponseMetadata> => {
  const formData = new FormData();

  // Add the file with proper metadata
  formData.append('file', file.buffer, {
    filename: file.originalname,
    contentType: file.mimetype,
  });
  const url = (await keyValueStoreService.get<string>(endpoint)) || '{}';

  const storageUrl = JSON.parse(url).storage.endpoint || defaultConfig.endpoint;

  // Add other required fields
  formData.append(
    'documentPath',
    `PipesHub/KnowledgeBase/private/${req.user?.userId}`,
  );
  formData.append('isVersionedFile', isVersionedFile.toString());
  formData.append('documentName', getFilenameWithoutExtension(documentName));

  try {
    const response = await axiosInstance.post(
      `${storageUrl}/api/v1/document/upload`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          Authorization: req.headers.authorization,
        },
      },
    );

    return {
      documentId: response.data?._id,
      documentName: response.data?.documentName,
    };
  } catch (error: any) {
    if (error.response.status === HTTP_STATUS.PERMANENT_REDIRECT) {
      const redirectUrl = error.response.headers.location;
      if (process.env.NODE_ENV == 'development') {
        logger.info('Redirecting to storage url', { redirectUrl });
      }

      // Extract document information from headers
      const documentId = error.response.headers['x-document-id'];
      const documentName = error.response.headers['x-document-name'];

      // Upload file in background (no callback - Python will handle Kafka events)
      axios({
        method: 'put',
        url: redirectUrl,
        data: file.buffer,
        headers: {
          'Content-Type': file.mimetype,
          'Content-Length': file.buffer.length,
        },
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
      }).catch((uploadError) => {
        logger.error('Background upload failed', {
          documentId,
          documentName,
          error: uploadError.message,
        });
      });

      return { documentId, documentName };
    } else {
      logger.error('Error uploading file to storage', {
        error: error.response.data,
      });
      throw error;
    }
  }
};

export const uploadNextVersionToStorage = async (
  req: AuthenticatedUserRequest,
  file: FileBufferInfo,
  documentId: string,
  keyValueStoreService: KeyValueStoreService,
  defaultConfig: DefaultStorageConfig,
): Promise<StorageResponseMetadata> => {
  const formData = new FormData();

  // Add the file with proper metadata
  formData.append('file', file.buffer, {
    filename: file.originalname,
    contentType: file.mimetype,
  });

  const url = (await keyValueStoreService.get<string>(endpoint)) || '{}';

  const storageUrl = JSON.parse(url).storage.endpoint || defaultConfig.endpoint;

  try {
    const response = await axiosInstance.post(
      `${storageUrl}/api/v1/document/${documentId}/uploadNextVersion`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          Authorization: req.headers.authorization,
        },
      },
    );

    return {
      documentId: response.data?._id,
      documentName: response.data?.documentName,
    };
  } catch (error: any) {
    logger.error('Error uploading file to storage', error.response.message);
    throw error;
  }
};

function getFilenameWithoutExtension(originalname: string) {
  const fileExtension = originalname.slice(originalname.lastIndexOf('.') + 1);
  return originalname.slice(0, -fileExtension.length - 1);
}
