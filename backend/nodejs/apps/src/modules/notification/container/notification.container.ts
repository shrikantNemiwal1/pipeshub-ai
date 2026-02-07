import { Container } from 'inversify';
import { NotificationService } from '../service/notification.service';
import { AuthTokenService } from '../../../libs/services/authtoken.service';
import { AppConfig } from '../../tokens_manager/config/config';
import { TYPES } from '../../../libs/types/container.types';

export class NotificationContainer {
  private static container: Container | null = null;

  static async initialize(appConfig: AppConfig): Promise<Container> {
    const container = new Container();
    const authTokenService = new AuthTokenService(
      appConfig.jwtSecret,
      appConfig.scopedJwtSecret,
    );
    container.bind<AuthTokenService>(TYPES.AuthTokenService).toConstantValue(authTokenService);
    container.bind(NotificationService).toSelf().inSingletonScope();
    this.container = container;
    return container;
  }

  static async dispose(): Promise<void> {
    if (this.container) {
      this.container.unbindAll();
    }
  }
}