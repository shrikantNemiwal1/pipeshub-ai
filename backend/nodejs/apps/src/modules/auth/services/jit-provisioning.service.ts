import { injectable, inject } from 'inversify';
import { Logger } from '../../../libs/services/logger.service';
import { BadRequestError, ServiceUnavailableError } from '../../../libs/errors/http.errors';
import { Users } from '../../user_management/schema/users.schema';
import { UserGroups } from '../../user_management/schema/userGroup.schema';
import { AppConfig } from '../../tokens_manager/config/config';
import axios from 'axios';
import { graphDbServiceJwtGenerator } from '../../../libs/utils/createJwt';

export interface JitUserDetails {
  firstName?: string;
  lastName?: string;
  fullName: string;
}

export type JitProvider = 'google' | 'microsoft' | 'azureAd' | 'oauth' | 'saml';

/**
 * Service responsible for Just-In-Time (JIT) user provisioning.
 * This service is shared across auth and user management modules to avoid circular dependencies.
 */
@injectable()
export class JitProvisioningService {
  constructor(
    @inject('Logger') private logger: Logger,
    @inject('AppConfig') private config: AppConfig,
  ) {}

  /**
   * Provision a new user via JIT from an OAuth/SAML provider.
   * Creates user, adds to everyone group, and publishes creation event.
   */
  async provisionUser(
    email: string,
    userDetails: JitUserDetails,
    orgId: string,
    provider: JitProvider,
  ) {
    this.logger.info(`Auto-provisioning user from ${provider}`, { email, orgId });

    // Check if user was previously deleted
    const deletedUser = await Users.findOne({
      email,
      orgId,
      isDeleted: true,
    });
    if (deletedUser) {
      throw new BadRequestError(
        'User account deleted by admin. Please contact your admin to restore your account.',
      );
    }

    const newUser = new Users({
      email,
      ...userDetails,
      orgId,
      hasLoggedIn: false,
      isDeleted: false,
    });

    await newUser.save();
    this.logger.info(`User created in MongoDB during ${provider} JIT provisioning`, { userId: newUser._id });

    // Add to everyone group
    await UserGroups.updateOne(
      { orgId, type: 'everyone', isDeleted: false },
      { $addToSet: { users: newUser._id } },
    );

    // Synchronously create user in graph database
    try {
      const graphToken = graphDbServiceJwtGenerator(
        orgId,
        this.config.scopedJwtSecret
      );
      
      await axios.post(
        `${this.config.connectorBackend}/api/v1/connectors/graph/user`,
        {
          userId: String(newUser._id),
          orgId: orgId,
          fullName: userDetails.fullName,
          email: email,
          firstName: userDetails.firstName,
          lastName: userDetails.lastName,
          syncAction: 'none', // Don't trigger immediate sync
        },
        {
          timeout: 30000, // Increased to 30s - KB creation takes time
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${graphToken}`,
          },
        }
      );
      this.logger.info(`User created in graph database with KB and permissions during ${provider} JIT provisioning`, { userId: newUser._id });
    } catch (graphError: any) {
      // Rollback: Delete user from MongoDB and remove from group
      this.logger.error(`Failed to create user in graph database during ${provider} JIT provisioning, rolling back`, {
        userId: newUser._id,
        error: graphError.message,
      });
      
      await Users.deleteOne({ _id: newUser._id });
      await UserGroups.updateOne(
        { orgId, type: 'everyone' },
        { $pull: { users: newUser._id } }
      );
      
      throw new ServiceUnavailableError(
        'Connector service is unavailable. Please try again later.'
      );
    }

    this.logger.info(`User auto-provisioned successfully via ${provider}`, {
      userId: newUser._id,
      email,
    });

    return newUser.toObject();
  }

  /**
   * Extract user details from Google ID token payload
   */
  extractGoogleUserDetails(payload: any, email: string): JitUserDetails {
    const firstName = payload?.given_name;
    const lastName = payload?.family_name;
    const displayName = payload?.name;

    const fullName =
      displayName ||
      [firstName, lastName].filter(Boolean).join(' ') ||
      email.split('@')[0];

    return {
      firstName: firstName || undefined,
      lastName: lastName || undefined,
      fullName,
    };
  }

  /**
   * Extract user details from Microsoft/Azure AD decoded token
   */
  extractMicrosoftUserDetails(decodedToken: any, email: string): JitUserDetails {
    const firstName = decodedToken?.given_name;
    const lastName = decodedToken?.family_name;
    const displayName = decodedToken?.name;

    const fullName =
      displayName ||
      [firstName, lastName].filter(Boolean).join(' ') ||
      email.split('@')[0];

    return {
      firstName: firstName || undefined,
      lastName: lastName || undefined,
      fullName,
    };
  }

  /**
   * Extract user details from OAuth userInfo response
   */
  extractOAuthUserDetails(userInfo: any, email: string): JitUserDetails {
    // Common OAuth/OIDC claims
    const firstName =
      userInfo?.given_name ||
      userInfo?.first_name ||
      userInfo?.firstName;
    const lastName =
      userInfo?.family_name ||
      userInfo?.last_name ||
      userInfo?.lastName;
    const displayName =
      userInfo?.name ||
      userInfo?.displayName ||
      userInfo?.preferred_username;

    const fullName =
      displayName ||
      [firstName, lastName].filter(Boolean).join(' ') ||
      email.split('@')[0];

    return {
      firstName: firstName || undefined,
      lastName: lastName || undefined,
      fullName,
    };
  }

  /**
   * Extract user details from SAML assertion with fallbacks for different IdP formats
   */
  extractSamlUserDetails(samlUser: any, email: string): JitUserDetails {
    const SAML_CLAIM_GIVENNAME =
      'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname';
    const SAML_OID_GIVENNAME = 'urn:oid:2.5.4.42';
    const SAML_CLAIM_SURNAME =
      'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname';
    const SAML_OID_SURNAME = 'urn:oid:2.5.4.4';
    const SAML_CLAIM_NAME =
      'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name';
    const SAML_OID_DISPLAYNAME = 'urn:oid:2.16.840.1.113730.3.1.241';

    // Try multiple SAML attribute names for first name
    const firstName =
      samlUser.firstName ||
      samlUser.givenName ||
      samlUser[SAML_CLAIM_GIVENNAME] ||
      samlUser[SAML_OID_GIVENNAME];

    // Try multiple SAML attribute names for last name
    const lastName =
      samlUser.lastName ||
      samlUser.surname ||
      samlUser.sn ||
      samlUser[SAML_CLAIM_SURNAME] ||
      samlUser[SAML_OID_SURNAME];

    // Try multiple SAML attribute names for display name
    const displayName =
      samlUser.displayName ||
      samlUser.name ||
      samlUser.fullName ||
      samlUser[SAML_CLAIM_NAME] ||
      samlUser[SAML_OID_DISPLAYNAME];

    // Construct full name with fallbacks
    const fullName =
      displayName ||
      [firstName, lastName].filter(Boolean).join(' ') ||
      email.split('@')[0];

    return {
      firstName: firstName || undefined,
      lastName: lastName || undefined,
      fullName,
    };
  }
}
