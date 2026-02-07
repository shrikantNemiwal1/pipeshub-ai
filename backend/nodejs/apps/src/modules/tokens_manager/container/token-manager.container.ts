import { Container } from 'inversify';
import { AppConfig, loadAppConfig } from '../config/config';
import { MongoService } from '../../../libs/services/mongo.service';
import { RedisService } from '../../../libs/services/redis.service';
import { Logger } from '../../../libs/services/logger.service';
import { ConfigurationManagerConfig } from '../../configuration_manager/config/config';
import { AuthTokenService } from '../../../libs/services/authtoken.service';
import { AuthMiddleware } from '../../../libs/middlewares/auth.middleware';
import { KeyValueStoreService } from '../../../libs/services/keyValueStore.service';

const loggerConfig = {
  service: 'Token Manager',
};

export class TokenManagerContainer {
  private static instance: Container;
  private static logger: Logger = Logger.getInstance(loggerConfig);

  static async initialize(
    configurationManagerConfig: ConfigurationManagerConfig,
  ): Promise<Container> {
    const container = new Container();
    const config: AppConfig = await loadAppConfig();
    // Bind configuration
    container
      .bind<AppConfig>('AppConfig')
      .toDynamicValue(() => config) // Always fetch latest reference
      .inTransientScope();

    // Bind logger
    container.bind<Logger>('Logger').toConstantValue(this.logger);
    container
      .bind<ConfigurationManagerConfig>('ConfigurationManagerConfig')
      .toConstantValue(configurationManagerConfig);
    // Initialize and bind services
    await this.initializeServices(container, config);

    this.instance = container;
    return container;
  }

  private static async initializeServices(
    container: Container,
    config: AppConfig,
  ): Promise<void> {
    try {
      const mongoService = new MongoService(config.mongo);
      await mongoService.initialize();
      container
        .bind<MongoService>('MongoService')
        .toConstantValue(mongoService);

      const redisService = new RedisService(
        config.redis,
        container.get('Logger'),
      );
      container
        .bind<RedisService>('RedisService')
        .toConstantValue(redisService);

      // Initialize KeyValueStoreService for PrometheusService dependency
      const configurationManagerConfig = container.get<ConfigurationManagerConfig>('ConfigurationManagerConfig');
      const keyValueStoreService = KeyValueStoreService.getInstance(
        configurationManagerConfig,
      );
      await keyValueStoreService.connect();
      container
        .bind<KeyValueStoreService>('KeyValueStoreService')
        .toConstantValue(keyValueStoreService);

      const jwtSecret = config.jwtSecret;
      const scopedJwtSecret = config.scopedJwtSecret;
      const authTokenService = new AuthTokenService(
        jwtSecret || ' ',
        scopedJwtSecret || ' ',
      );
      const authMiddleware = new AuthMiddleware(
        container.get('Logger'),
        authTokenService,
      );
      container
        .bind<AuthMiddleware>('AuthMiddleware')
        .toConstantValue(authMiddleware);
    } catch (error) {
      const logger = container.get<Logger>('Logger');
      logger.error('Failed to initialize services', {
        error: error instanceof Error ? error.message : 'Unknown error',
      });
      throw error;
    }
  }

  static getInstance(): Container {
    if (!this.instance) {
      throw new Error('Service container not initialized');
    }
    return this.instance;
  }

  static async dispose(): Promise<void> {
    if (this.instance) {
      try {
        // Get specific services that need to be disconnected
        const mongoService = this.instance.isBound('MongoService')
          ? this.instance.get<MongoService>('MongoService')
          : null;

        const redisService = this.instance.isBound('RedisService')
          ? this.instance.get<RedisService>('RedisService')
          : null;

        // Disconnect services if they have a disconnect method

        if (redisService && redisService.isConnected()) {
          await redisService.disconnect();
        }
        if (mongoService && mongoService.isConnected()) {
          await mongoService.destroy();
        }

        this.logger.info('All services disconnected successfully');
      } catch (error) {
        this.logger.error('Error while disconnecting services', {
          error: error instanceof Error ? error.message : 'Unknown error',
        });
      } finally {
        this.instance = null!;
      }
    }
  }
}
